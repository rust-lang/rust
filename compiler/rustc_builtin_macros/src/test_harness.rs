// Code that generates a test runner to run all the tests in a crate

use std::mem;

use rustc_ast as ast;
use rustc_ast::entry::EntryPointType;
use rustc_ast::mut_visit::*;
use rustc_ast::ptr::P;
use rustc_ast::visit::{Visitor, walk_item};
use rustc_ast::{ModKind, attr};
use rustc_errors::DiagCtxtHandle;
use rustc_expand::base::{ExtCtxt, ResolverExpand};
use rustc_expand::expand::{AstFragment, ExpansionConfig};
use rustc_feature::Features;
use rustc_lint_defs::BuiltinLintDiag;
use rustc_session::Session;
use rustc_session::lint::builtin::UNNAMEABLE_TEST_ITEMS;
use rustc_span::hygiene::{AstPass, SyntaxContext, Transparency};
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, sym};
use rustc_target::spec::PanicStrategy;
use smallvec::smallvec;
use thin_vec::{ThinVec, thin_vec};
use tracing::debug;

use crate::errors;

#[derive(Clone)]
struct Test {
    span: Span,
    ident: Ident,
    name: Symbol,
}

struct TestCtxt<'a> {
    ext_cx: ExtCtxt<'a>,
    panic_strategy: PanicStrategy,
    def_site: Span,
    test_cases: Vec<Test>,
    reexport_test_harness_main: Option<Symbol>,
    test_runner: Option<ast::Path>,
}

/// Traverse the crate, collecting all the test functions, eliding any
/// existing main functions, and synthesizing a main test harness
pub fn inject(
    krate: &mut ast::Crate,
    sess: &Session,
    features: &Features,
    resolver: &mut dyn ResolverExpand,
) {
    let dcx = sess.dcx();
    let panic_strategy = sess.panic_strategy();
    let platform_panic_strategy = sess.target.panic_strategy;

    // Check for #![reexport_test_harness_main = "some_name"] which gives the
    // main test function the name `some_name` without hygiene. This needs to be
    // unconditional, so that the attribute is still marked as used in
    // non-test builds.
    let reexport_test_harness_main =
        attr::first_attr_value_str_by_name(&krate.attrs, sym::reexport_test_harness_main);

    // Do this here so that the test_runner crate attribute gets marked as used
    // even in non-test builds
    let test_runner = get_test_runner(dcx, krate);

    if sess.is_test_crate() {
        let panic_strategy = match (panic_strategy, sess.opts.unstable_opts.panic_abort_tests) {
            (PanicStrategy::Abort, true) => PanicStrategy::Abort,
            (PanicStrategy::Abort, false) => {
                if panic_strategy == platform_panic_strategy {
                    // Silently allow compiling with panic=abort on these platforms,
                    // but with old behavior (abort if a test fails).
                } else {
                    dcx.emit_err(errors::TestsNotSupport {});
                }
                PanicStrategy::Unwind
            }
            (PanicStrategy::Unwind, _) => PanicStrategy::Unwind,
        };
        generate_test_harness(
            sess,
            resolver,
            reexport_test_harness_main,
            krate,
            features,
            panic_strategy,
            test_runner,
        )
    }
}

struct TestHarnessGenerator<'a> {
    cx: TestCtxt<'a>,
    tests: Vec<Test>,
}

impl TestHarnessGenerator<'_> {
    fn add_test_cases(&mut self, node_id: ast::NodeId, span: Span, prev_tests: Vec<Test>) {
        let mut tests = mem::replace(&mut self.tests, prev_tests);

        if !tests.is_empty() {
            // Create an identifier that will hygienically resolve the test
            // case name, even in another module.
            let expn_id = self.cx.ext_cx.resolver.expansion_for_ast_pass(
                span,
                AstPass::TestHarness,
                &[],
                Some(node_id),
            );
            for test in &mut tests {
                // See the comment on `mk_main` for why we're using
                // `apply_mark` directly.
                test.ident.span =
                    test.ident.span.apply_mark(expn_id.to_expn_id(), Transparency::Opaque);
            }
            self.cx.test_cases.extend(tests);
        }
    }
}

impl<'a> MutVisitor for TestHarnessGenerator<'a> {
    fn visit_crate(&mut self, c: &mut ast::Crate) {
        let prev_tests = mem::take(&mut self.tests);
        walk_crate(self, c);
        self.add_test_cases(ast::CRATE_NODE_ID, c.spans.inner_span, prev_tests);

        // Create a main function to run our tests
        c.items.push(mk_main(&mut self.cx));
    }

    fn visit_item(&mut self, item: &mut ast::Item) {
        if let Some(name) = get_test_name(&item) {
            debug!("this is a test item");

            // `unwrap` is ok because only functions, consts, and static should reach here.
            let test = Test { span: item.span, ident: item.kind.ident().unwrap(), name };
            self.tests.push(test);
        }

        // We don't want to recurse into anything other than mods, since
        // mods or tests inside of functions will break things
        if let ast::ItemKind::Mod(
            _,
            _,
            ModKind::Loaded(.., ast::ModSpans { inner_span: span, .. }, _),
        ) = item.kind
        {
            let prev_tests = mem::take(&mut self.tests);
            walk_item_kind(&mut item.kind, item.span, item.id, &mut item.vis, (), self);
            self.add_test_cases(item.id, span, prev_tests);
        } else {
            // But in those cases, we emit a lint to warn the user of these missing tests.
            walk_item(&mut InnerItemLinter { sess: self.cx.ext_cx.sess }, &item);
        }
    }
}

struct InnerItemLinter<'a> {
    sess: &'a Session,
}

impl<'a> Visitor<'a> for InnerItemLinter<'_> {
    fn visit_item(&mut self, i: &'a ast::Item) {
        if let Some(attr) = attr::find_by_name(&i.attrs, sym::rustc_test_marker) {
            self.sess.psess.buffer_lint(
                UNNAMEABLE_TEST_ITEMS,
                attr.span,
                i.id,
                BuiltinLintDiag::UnnameableTestItems,
            );
        }
    }
}

fn entry_point_type(item: &ast::Item, at_root: bool) -> EntryPointType {
    match &item.kind {
        ast::ItemKind::Fn(fn_) => {
            rustc_ast::entry::entry_point_type(&item.attrs, at_root, Some(fn_.ident.name))
        }
        _ => EntryPointType::None,
    }
}

/// A folder used to remove any entry points (like fn main) because the harness
/// coroutine will provide its own
struct EntryPointCleaner<'a> {
    // Current depth in the ast
    sess: &'a Session,
    depth: usize,
    def_site: Span,
}

impl<'a> MutVisitor for EntryPointCleaner<'a> {
    fn visit_item(&mut self, item: &mut ast::Item) {
        self.depth += 1;
        ast::mut_visit::walk_item(self, item);
        self.depth -= 1;

        // Remove any #[rustc_main] from the AST so it doesn't
        // clash with the one we're going to add, but mark it as
        // #[allow(dead_code)] to avoid printing warnings.
        match entry_point_type(&item, self.depth == 0) {
            EntryPointType::MainNamed | EntryPointType::RustcMainAttr => {
                let allow_dead_code = attr::mk_attr_nested_word(
                    &self.sess.psess.attr_id_generator,
                    ast::AttrStyle::Outer,
                    ast::Safety::Default,
                    sym::allow,
                    sym::dead_code,
                    self.def_site,
                );
                item.attrs.retain(|attr| !attr.has_name(sym::rustc_main));
                item.attrs.push(allow_dead_code);
            }
            EntryPointType::None | EntryPointType::OtherMain => {}
        };
    }
}

/// Crawl over the crate, inserting test reexports and the test main function
fn generate_test_harness(
    sess: &Session,
    resolver: &mut dyn ResolverExpand,
    reexport_test_harness_main: Option<Symbol>,
    krate: &mut ast::Crate,
    features: &Features,
    panic_strategy: PanicStrategy,
    test_runner: Option<ast::Path>,
) {
    let econfig = ExpansionConfig::default("test".to_string(), features);
    let ext_cx = ExtCtxt::new(sess, econfig, resolver, None);

    let expn_id = ext_cx.resolver.expansion_for_ast_pass(
        DUMMY_SP,
        AstPass::TestHarness,
        &[sym::test, sym::rustc_attrs, sym::coverage_attribute],
        None,
    );
    let def_site = DUMMY_SP.with_def_site_ctxt(expn_id.to_expn_id());

    // Remove the entry points
    let mut cleaner = EntryPointCleaner { sess, depth: 0, def_site };
    cleaner.visit_crate(krate);

    let cx = TestCtxt {
        ext_cx,
        panic_strategy,
        def_site,
        test_cases: Vec::new(),
        reexport_test_harness_main,
        test_runner,
    };

    TestHarnessGenerator { cx, tests: Vec::new() }.visit_crate(krate);
}

/// Creates a function item for use as the main function of a test build.
/// This function will call the `test_runner` as specified by the crate attribute
///
/// By default this expands to
///
/// ```ignore (messes with test internals)
/// #[rustc_main]
/// pub fn main() {
///     extern crate test;
///     test::test_main_static(&[
///         &test_const1,
///         &test_const2,
///         &test_const3,
///     ]);
/// }
/// ```
///
/// Most of the Ident have the usual def-site hygiene for the AST pass. The
/// exception is the `test_const`s. These have a syntax context that has two
/// opaque marks: one from the expansion of `test` or `test_case`, and one
/// generated  in `TestHarnessGenerator::visit_item`. When resolving this
/// identifier after failing to find a matching identifier in the root module
/// we remove the outer mark, and try resolving at its def-site, which will
/// then resolve to `test_const`.
///
/// The expansion here can be controlled by two attributes:
///
/// [`TestCtxt::reexport_test_harness_main`] provides a different name for the `main`
/// function and [`TestCtxt::test_runner`] provides a path that replaces
/// `test::test_main_static`.
fn mk_main(cx: &mut TestCtxt<'_>) -> P<ast::Item> {
    let sp = cx.def_site;
    let ecx = &cx.ext_cx;
    let test_ident = Ident::new(sym::test, sp);

    let runner_name = match cx.panic_strategy {
        PanicStrategy::Unwind => "test_main_static",
        PanicStrategy::Abort => "test_main_static_abort",
    };

    // test::test_main_static(...)
    let mut test_runner = cx.test_runner.clone().unwrap_or_else(|| {
        ecx.path(sp, vec![test_ident, Ident::from_str_and_span(runner_name, sp)])
    });

    test_runner.span = sp;

    let test_main_path_expr = ecx.expr_path(test_runner);
    let call_test_main = ecx.expr_call(sp, test_main_path_expr, thin_vec![mk_tests_slice(cx, sp)]);
    let call_test_main = ecx.stmt_expr(call_test_main);

    // extern crate test
    let test_extern_stmt = ecx.stmt_item(
        sp,
        ecx.item(sp, ast::AttrVec::new(), ast::ItemKind::ExternCrate(None, test_ident)),
    );

    // #[rustc_main]
    let main_attr = ecx.attr_word(sym::rustc_main, sp);
    // #[coverage(off)]
    let coverage_attr = ecx.attr_nested_word(sym::coverage, sym::off, sp);
    // #[doc(hidden)]
    let doc_hidden_attr = ecx.attr_nested_word(sym::doc, sym::hidden, sp);

    // pub fn main() { ... }
    let main_ret_ty = ecx.ty(sp, ast::TyKind::Tup(ThinVec::new()));

    // If no test runner is provided we need to import the test crate
    let main_body = if cx.test_runner.is_none() {
        ecx.block(sp, thin_vec![test_extern_stmt, call_test_main])
    } else {
        ecx.block(sp, thin_vec![call_test_main])
    };

    let decl = ecx.fn_decl(ThinVec::new(), ast::FnRetTy::Ty(main_ret_ty));
    let sig = ast::FnSig { decl, header: ast::FnHeader::default(), span: sp };
    let defaultness = ast::Defaultness::Final;

    // Honor the reexport_test_harness_main attribute
    let main_ident = match cx.reexport_test_harness_main {
        Some(sym) => Ident::new(sym, sp.with_ctxt(SyntaxContext::root())),
        None => Ident::new(sym::main, sp),
    };

    let main = ast::ItemKind::Fn(Box::new(ast::Fn {
        defaultness,
        sig,
        ident: main_ident,
        generics: ast::Generics::default(),
        contract: None,
        body: Some(main_body),
        define_opaque: None,
    }));

    let main = P(ast::Item {
        attrs: thin_vec![main_attr, coverage_attr, doc_hidden_attr],
        id: ast::DUMMY_NODE_ID,
        kind: main,
        vis: ast::Visibility { span: sp, kind: ast::VisibilityKind::Public, tokens: None },
        span: sp,
        tokens: None,
    });

    // Integrate the new item into existing module structures.
    let main = AstFragment::Items(smallvec![main]);
    cx.ext_cx.monotonic_expander().fully_expand_fragment(main).make_items().pop().unwrap()
}

/// Creates a slice containing every test like so:
/// &[&test1, &test2]
fn mk_tests_slice(cx: &TestCtxt<'_>, sp: Span) -> P<ast::Expr> {
    debug!("building test vector from {} tests", cx.test_cases.len());
    let ecx = &cx.ext_cx;

    let mut tests = cx.test_cases.clone();
    tests.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));

    ecx.expr_array_ref(
        sp,
        tests
            .iter()
            .map(|test| {
                ecx.expr_addr_of(test.span, ecx.expr_path(ecx.path(test.span, vec![test.ident])))
            })
            .collect(),
    )
}

fn get_test_name(i: &ast::Item) -> Option<Symbol> {
    attr::first_attr_value_str_by_name(&i.attrs, sym::rustc_test_marker)
}

fn get_test_runner(dcx: DiagCtxtHandle<'_>, krate: &ast::Crate) -> Option<ast::Path> {
    let test_attr = attr::find_by_name(&krate.attrs, sym::test_runner)?;
    let meta_list = test_attr.meta_item_list()?;
    let span = test_attr.span;
    match &*meta_list {
        [single] => match single.meta_item() {
            Some(meta_item) if meta_item.is_word() => return Some(meta_item.path.clone()),
            _ => {
                dcx.emit_err(errors::TestRunnerInvalid { span });
            }
        },
        _ => {
            dcx.emit_err(errors::TestRunnerNargs { span });
        }
    }
    None
}
