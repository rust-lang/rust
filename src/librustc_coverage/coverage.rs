//! Injects code coverage instrumentation into the AST.

use rustc::util::common::ErrorReported;
use rustc_ast::ast::*;
use rustc_ast::mut_visit::{self, MutVisitor};
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_resolve::Resolver;
use rustc_span::symbol::sym;
use rustc_span::symbol::Symbol;
use rustc_span::Span;
use smallvec::SmallVec;

use log::trace;
use std::ops::DerefMut;
use std::sync::Mutex;

pub type Result<T> = std::result::Result<T, ErrorReported>;

struct CoverageRegion {
    region: Span,
    counter_hash: u128,
}

static mut COVERAGE_REGIONS: Option<Mutex<Vec<CoverageRegion>>> = None;

impl CoverageRegion {
    /// Generates a unique coverage region identifier to associate with
    /// a counter. The counter increment statement is injected into the
    /// code at the start of a coverage region.
    // FIXME(richkadel): This function will need additional arguments and/or context
    // data from which to generate the hash values.
    fn generate_hash(region: &Span) -> u128 {
        // THIS IS NOT THREAD SAFE, BUT WILL BE REPLACED WITH HASH FUNCTION ANYWAY.
        // Seems like lazy_static is not used in the compiler at all.
        static mut NEXT_COUNTER_ID: Option<Mutex<u128>> = None;
        let counter_hash = {
            let counter_id = unsafe {
                &match NEXT_COUNTER_ID.as_ref() {
                    Some(counter_id) => counter_id,
                    None => {
                        NEXT_COUNTER_ID = Some(Mutex::new(0));
                        NEXT_COUNTER_ID.as_ref().unwrap()
                    }
                }
            };
            let mut locked_counter_id = counter_id.lock().unwrap();
            *locked_counter_id += 1;
            *locked_counter_id
        };

        let coverage_regions = unsafe {
            &match COVERAGE_REGIONS.as_ref() {
                Some(coverage_regions) => coverage_regions,
                None => {
                    COVERAGE_REGIONS = Some(Mutex::new(vec![]));
                    COVERAGE_REGIONS.as_ref().unwrap()
                }
            }
        };
        let mut locked_coverage_regions = coverage_regions.lock().unwrap();
        locked_coverage_regions.push(CoverageRegion { region: region.clone(), counter_hash });

        // return the counter hash value
        counter_hash
    }

    pub fn write_coverage_regions(/* filename param? */) {
        unsafe {
            if let Some(coverage_regions) = COVERAGE_REGIONS.as_ref() {
                let locked_coverage_regions = coverage_regions.lock().unwrap();
                for coverage in locked_coverage_regions.iter() {
                    println!("{}: {:?}", coverage.counter_hash, coverage.region);
                }
            }
        }
    }
}

struct CoverageInjector<'res, 'internal> {
    resolver: &'res mut Resolver<'internal>,
    span: Span,
}

impl CoverageInjector<'_, '_> {
    fn at<'res, 'internal>(
        resolver: &'res mut Resolver<'internal>,
        span: Span,
    ) -> CoverageInjector<'res, 'internal> {
        CoverageInjector { resolver, span }
    }

    fn next_ast_node_id(&mut self) -> NodeId {
        self.resolver.next_node_id()
    }

    fn expr(&mut self, kind: ExprKind, span: Span) -> P<Expr> {
        P(Expr { kind, span, attrs: AttrVec::new(), id: self.next_ast_node_id() })
    }

    fn path_segment(&mut self, string: &str) -> PathSegment {
        PathSegment {
            ident: Ident::from_str_and_span(string, self.span.shrink_to_lo()),
            id: self.next_ast_node_id(),
            args: None,
        }
    }

    fn coverage_count_fn_path(&mut self, print_coverage_report: bool) -> P<Expr> {
        let fn_name = if print_coverage_report { "count_and_report" } else { "count" };
        let path = Path {
            span: self.span.shrink_to_lo(),
            segments: vec![
                self.path_segment("std"),
                self.path_segment("coverage"),
                self.path_segment(fn_name),
            ],
        };
        self.expr(ExprKind::Path(None, path), self.span.shrink_to_lo())
    }

    fn coverage_counter_hash_lit(&mut self, counter_hash: u128) -> P<Expr> {
        let token =
            token::Lit::new(token::Integer, sym::integer(counter_hash), /*suffix=*/ None);
        let kind = LitKind::Int(counter_hash, LitIntType::Unsuffixed);
        let lit = Lit { token, kind, span: self.span.shrink_to_lo() };
        self.expr(ExprKind::Lit(lit), self.span.shrink_to_lo())
    }

    fn call(&mut self, fn_path: P<Expr>, args: Vec<P<Expr>>) -> P<Expr> {
        self.expr(ExprKind::Call(fn_path, args), self.span.clone())
    }
}

struct CoverageVisitor<'res, 'internal> {
    resolver: &'res mut Resolver<'internal>,
    function_stack: Vec<Symbol>,
    main_block_id: Option<NodeId>,
}

impl CoverageVisitor<'_, '_> {
    fn new<'res, 'internal>(
        resolver: &'res mut Resolver<'internal>,
    ) -> CoverageVisitor<'res, 'internal> {
        CoverageVisitor { resolver, function_stack: vec![], main_block_id: None }
    }

    fn next_ast_node_id(&mut self) -> NodeId {
        self.resolver.next_node_id()
    }

    fn is_visiting_main(&self) -> bool {
        if let Some(current_fn) = self.function_stack.last() {
            *current_fn == sym::main
        } else {
            false
        }
    }

    fn empty_tuple(&mut self, span: Span) -> P<Expr> {
        P(Expr {
            kind: ExprKind::Tup(vec![]),
            span,
            attrs: AttrVec::new(),
            id: self.next_ast_node_id(),
        })
    }

    fn wrap_and_count_expr(
        &mut self,
        coverage_span: &Span,
        wrapped_expr: P<Expr>,
        print_coverage_report: bool,
    ) -> P<Expr> {
        let mut injector = CoverageInjector::at(&mut self.resolver, wrapped_expr.span.clone());
        let counter_hash = CoverageRegion::generate_hash(coverage_span);
        let coverage_count_fn = injector.coverage_count_fn_path(print_coverage_report);
        let args = vec![injector.coverage_counter_hash_lit(counter_hash), wrapped_expr];
        injector.call(coverage_count_fn, args)
    }

    fn wrap_and_count_stmt(
        &mut self,
        coverage_span: &Span,
        wrapped_expr: P<Expr>,
        print_coverage_report: bool,
    ) -> Stmt {
        Stmt {
            id: self.next_ast_node_id(),
            span: wrapped_expr.span.clone(),
            kind: StmtKind::Semi(self.wrap_and_count_expr(
                coverage_span,
                wrapped_expr,
                print_coverage_report,
            )),
        }
    }

    fn count_stmt(
        &mut self,
        coverage_span: &Span,
        inject_site: Span,
        print_coverage_report: bool,
    ) -> Stmt {
        let empty_tuple = self.empty_tuple(inject_site);
        self.wrap_and_count_stmt(coverage_span, empty_tuple, print_coverage_report)
    }

    fn instrument_block(&mut self, block: &mut Block) {
        trace!("instrument_block: {:?}", block);
        if let Some(mut last) = block.stmts.pop() {
            let mut report = false;
            if let Some(main) = self.main_block_id {
                report = block.id == main
            }

            match &mut last.kind {
                StmtKind::Expr(result_expr) => {
                    let wrapped_expr = result_expr.clone();
                    *result_expr = self.wrap_and_count_expr(&block.span, wrapped_expr, report);
                    block.stmts.push(last);
                }
                StmtKind::Semi(expr) => {
                    if let ExprKind::Ret(..) = expr.kind {
                        report = self.is_visiting_main();
                    }

                    match &mut expr.deref_mut().kind {
                        ExprKind::Break(_, result_expr)
                        | ExprKind::Ret(result_expr)
                        | ExprKind::Yield(result_expr) => {
                            match result_expr.take() {
                                Some(wrapped_expr) => {
                                    *result_expr = Some(self.wrap_and_count_expr(
                                        &block.span,
                                        wrapped_expr,
                                        report,
                                    ));
                                }
                                None => {
                                    block.stmts.push(self.count_stmt(
                                        &block.span,
                                        last.span.shrink_to_lo(),
                                        report,
                                    ));
                                }
                            }
                            block.stmts.push(last);
                        }
                        ExprKind::Continue(..) => {
                            block.stmts.push(self.count_stmt(
                                &block.span,
                                last.span.shrink_to_lo(),
                                report,
                            ));
                            block.stmts.push(last);
                        }
                        _ => {
                            let insert_after_last = last.span.shrink_to_hi();
                            block.stmts.push(last);
                            block.stmts.push(self.count_stmt(
                                &block.span,
                                insert_after_last,
                                report,
                            ));
                        }
                    }
                }
                _ => (),
            }
        }
    }
}

impl MutVisitor for CoverageVisitor<'_, '_> {
    fn visit_block(&mut self, block: &mut P<Block>) {
        self.instrument_block(block.deref_mut());
        mut_visit::noop_visit_block(block, self);
    }

    fn flat_map_item(&mut self, item: P<Item>) -> SmallVec<[P<Item>; 1]> {
        if let ItemKind::Fn(_defaultness, _signature, _generics, block) = &item.kind {
            if item.ident.name == sym::main {
                if let Some(block) = block {
                    self.main_block_id = Some(block.id);
                }
            }
            self.function_stack.push(item.ident.name);
            let result = mut_visit::noop_flat_map_item(item, self);
            self.function_stack.pop();
            result
        } else {
            mut_visit::noop_flat_map_item(item, self)
        }
    }

    // FIXME(richkadel):
    // add more visit_???() functions for language constructs that are branched statements without
    // blocks, such as:
    //    visit match arm expr
    // if not block, wrap expr in block and inject counter
    //
    // There are language constructs that have statement spans that don't require
    // braces if only one statement, in which case, they PROBABLY don't hit "Block", and
    // therefore, I need to insert the counters in other parts of the AST as well, while
    // also virtually inserting the curly braces:
    //   * closures: |...| stmt  ->   |...| { coverage::counter(n); stmt }
    //   * match arms: match variant { pat => stmt, ...} -> match variant { pat => { coverage::counter(n); stmt } ... }
    // Lazy boolean operators: logical expressions that may not be executed if prior expressions obviate the need
    //     "the right-hand operand is only evaluated when the left-hand operand does not already
    //      determine the result of the expression. That is, || only evaluates its right-hand
    //      operand when the left-hand operand evaluates to false, and && only when it evaluates to
    //      true."
    //     * if true || stmt   ->   if true || coverage::wrap_and_count(n, { stmt } )
    //     make sure operator precedence is handled with boolean expressions
    //        (?? IS THAT if false && coverage::count(condition2 && coverage::count(condition3)))
    //        I THINK it doesn't matter if the operator is && or ||
    //        Unless there are parentheses, each operator requires a coverage wrap_and_count around
    //        ALL remaining && or || conditions on the right side, and then recursively nested.
    //        Rust calls parentheticals "Grouped expressions"
    // `?` operator which can invoke the equivalent of an early return (of `Err(...)`).
    //   * {expr_possibly_in_block}?   ->   coverage::counter(n, {expr_possibly_in_block})?
    // Any others?
    //   * NOT let var: type = expr (since there's no branching to worry about)
    //   * NOT for or while loop expressions because they aren't optionally executed

    // FIXME(richkadel): if multi-threaded, are we assured that exiting main means all threads have completed?
}

pub fn instrument<'res>(mut krate: Crate, resolver: &'res mut Resolver<'_>) -> Result<Crate> {
    trace!("Calling coverage::instrument() for {:?}", &krate);
    mut_visit::noop_visit_crate(&mut krate, &mut CoverageVisitor::new(resolver));
    CoverageRegion::write_coverage_regions();
    Ok(krate)
}
