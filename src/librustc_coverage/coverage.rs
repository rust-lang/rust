//! Injects code coverage instrumentation into the AST.

use rustc::util::common::ErrorReported;
use rustc_ast::ast::*;
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::mut_visit::{self, MutVisitor};
use rustc_resolve::Resolver;
use rustc_span::symbol::sym;
use rustc_span::Span;

use log::{debug, trace};
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
    // TODO(richkadel): This function will need additional arguments and/or context
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
        locked_coverage_regions.push(CoverageRegion {
            region: region.clone(),
            counter_hash,
        });

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

struct CoverageInjector<'res,'internal> {
    resolver: &'res mut Resolver<'internal>,
    span: Span,
}

impl CoverageInjector<'_,'_> {
    fn at<'res,'internal>(resolver: &'res mut Resolver<'internal>, span: Span) -> CoverageInjector<'res,'internal> {
        CoverageInjector {
            resolver,
            span,
        }
    }

    fn next_ast_node_id(&mut self) -> NodeId {
        self.resolver.next_node_id()
    }

    fn span(&self) -> Span {
        self.span.clone()
    }

    fn expr(&mut self, kind: ExprKind) -> P<Expr> {
        P(Expr {
            kind,
            span: self.span(),
            attrs: AttrVec::new(),
            id: self.next_ast_node_id(),
        })
    }

    fn path_segment(&mut self, string: &str) -> PathSegment {
        PathSegment {
            ident: Ident::from_str_and_span(string, self.span()),
            id: self.next_ast_node_id(),
            args: None,
        }
    }

    fn coverage_count_fn_path(&mut self) -> P<Expr> {
        let path = Path {
            span: self.span(),
            segments: vec![
                self.path_segment("coverage"),
                self.path_segment("count"),
            ],
        };
        self.expr(ExprKind::Path(None, path))
    }

    fn coverage_counter_hash_lit(&mut self, counter_hash: u128) -> P<Expr> {
        let token = token::Lit::new(token::Integer, sym::integer(counter_hash), /*suffix=*/None);
        let kind = LitKind::Int(
            counter_hash,
            LitIntType::Unsigned(UintTy::U128), // TODO: this should not be necessary (should be "Unsuffixed" per JSON)
        );
        let lit = Lit { token, kind, span: self.span() };
        self.expr(ExprKind::Lit(lit))
    }

    fn call(&mut self, fn_path: P<Expr>, args: Vec<P<Expr>>) -> P<Expr> {
        self.expr(ExprKind::Call(fn_path, args))
    }

    fn counter_stmt(&mut self, coverage_span: &Span) -> Stmt {
        let counter_hash = CoverageRegion::generate_hash(coverage_span);
        let coverage_count_fn = self.coverage_count_fn_path();
        let args = vec![ self.coverage_counter_hash_lit(counter_hash) ];
        let call = self.call(coverage_count_fn, args);

        Stmt {
            id: self.next_ast_node_id(),
            span: self.span(),
            kind: StmtKind::Semi(call)
        }
    }
}

struct CoverageVisitor<'res,'internal> {
    resolver: &'res mut Resolver<'internal>,
}

impl CoverageVisitor<'_,'_> {

    fn instrument_block(&mut self, block: &mut Block) {
        trace!("instrument_block: {:?}", block);
        let _ = self.resolver;
        if let Some(last) = block.stmts.last() {
            let inject_site = if last.is_expr() {
                last.span.shrink_to_lo()
            } else {
                last.span.shrink_to_hi()
            };

            let mut coverage_injector = CoverageInjector::at(&mut self.resolver, inject_site);
            let counter_stmt = coverage_injector.counter_stmt(&block.span);
            if last.is_expr() {
                block.stmts.insert(block.stmts.len()-1, counter_stmt);
            } else {
                block.stmts.push(counter_stmt);
            }

            // TODO(richkadel): The span should stop at the first occurrence of an early return
            // (if any), and if there is one, a new counter should be inserted just after the branch
            // that ended in the early return (or panic?), and a new span should start from just after that
            // injected counter, and ending at the end of the block, or at another early return if
            // there is another.

            // FIRST DEMO VERSION SHOULD WORK FOR BASIC BLOCKS
            // THAT DON'T `break`, `continue`, `return`, or panic in any nested
            // branches. For those, we need to add additional counters beyond the code that might be
            // skipped.
            //
            // IMPORTANT! BREAK UP BLOCKS THAT MAY PANIC, SIMILAR TO EARLY RETURN, OR INJECT COUNTER
            // AFTER THE LAST `Semi` Stmt, AND NOTE PANICS IN SOME OTHER WAY?

            // There may be existing checks of similar types. See for instance "targeted_by_break"
            // in librustc_ast_lowering/src/lib.rs
        }
    }
}

impl MutVisitor for CoverageVisitor<'_,'_> {

    fn visit_block(&mut self, block: &mut P<Block>) {
        self.instrument_block(block.deref_mut());
        mut_visit::noop_visit_block(block, self);
    }

    // TODO(richkadel): visit match arm expr (if not block, wrap expr in block and inject counter)
    // ALSO! There are language constructs that have statement spans that don't require
    // braces if only one statement, in which case, they PROBABLY don't hit "Block", and
    // therefore, I need to insert the counters in other parts of the AST as well, while
    // also virtually inserting the curly braces:
    //   * closures: |...| stmt  ->   |...| { coverage::counter(n); stmt }
    //   * match arms: match variant { pat => stmt, ...} -> match variant { pat => { coverage::counter(n); stmt } ... }
    // any others?
}

pub fn instrument<'res>(krate: &mut Crate, resolver: &'res mut Resolver<'_>) { //, resolver: &'a mut Resolver<'a>) {
    debug!("Calling instrument for {:?}", krate);
    mut_visit::noop_visit_crate(krate, &mut CoverageVisitor { resolver } ); // , &mut CoverageVisitor::new(resolver));
    CoverageRegion::write_coverage_regions();
}