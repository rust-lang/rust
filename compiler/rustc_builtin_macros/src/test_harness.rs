// Code that generates a test runner to run all the tests in a crate

use rustc_ast as ast;
use rustc_ast::attr;
use rustc_ast::entry::EntryPointType;
use rustc_ast::mut_visit::{ExpectOne, *};
use rustc_ast::ptr::P;
use rustc_expand::base::{ExtCtxt, ResolverExpand};
use rustc_expand::expand::{AstFragment, ExpansionConfig};
use rustc_feature::Features;
use rustc_session::Session;
use rustc_span::hygiene::{AstPass, SyntaxContext, Transparency};
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::spec::PanicStrategy;
use smallvec::{smallvec, SmallVec};
use tracing::debug;

use std::{iter, mem};

struct Test {
    span: Span,
    ident: Ident,
}

struct TestCtxt<'a> {
    ext_cx: ExtCtxt<'a>,
    panic_strategy: PanicStrategy,
    def_site: Span,
    test_cases: Vec<Test>,
    reexport_test_harness_main: Option<Symbol>,
    test_runner: Option<ast::Path>,
}

// Traverse the crate, collecting all the test functions, eliding any
// existing main functions, and synthesizing a main test harness
pub fn inject(sess: &Session, resolver: &mut dyn ResolverExpand, krate: &mut ast::Crate) {
    let span_diagnostic = sess.diagnostic();
    let panic_strategy = sess.panic_strategy();
    let platform_panic_strategy = sess.target.options.panic_strategy;

    // Check for #![reexport_test_harness_main = "some_name"] which gives the
    // main test function the name `some_name` without hygiene. This needs to be
    // unconditional, so that the attribute is still marked as used in
    // non-test builds.
    let reexport_test_harness_main =
        sess.first_attr_value_str_by_name(&krate.attrs, sym::reexport_test_harness_main);

    // Do this here so that the test_runner crate attribute gets marked as used
    // even in non-test builds
    let test_runner = get_test_runner(sess, span_diagnostic, &krate);

    if sess.opts.test {
        let panic_strategy = match (panic_strategy, sess.opts.debugging_opts.panic_abort_tests) {
            (PanicStrategy::Abort, true) => PanicStrategy::Abort,
            (PanicStrategy::Abort, false) => {
                if panic_strategy == platform_panic_strategy {
                    // Silently allow compiling with panic=abort on these platforms,
                    // but with old behavior (abort if a test fails).
                } else {
                    span_diagnostic.err(
                        "building tests with panic=abort is not supported \
                                         without `-Zpanic_abort_tests`",
                    );
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
            &sess.features_untracked(),
            panic_strategy,
            test_runner,
        )
    }
}

struct TestHarnessGenerator<'a> {
    cx: TestCtxt<'a>,
    tests: Vec<Test>,
}

impl<'a> MutVisitor for TestHarnessGenerator<'a> {
    fn visit_crate(&mut self, c: &mut ast::Crate) {
        noop_visit_crate(c, self);

        // Create a main function to run our tests
        c.module.items.push(mk_main(&mut self.cx));
    }

    fn flat_map_item(&mut self, i: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        let mut item = i.into_inner();
        if is_test_case(&self.cx.ext_cx.sess, &item) {
            debug!("this is a test item");

            let test = Test { span: item.span, ident: item.ident };
            self.tests.push(test);
        }

        // We don't want to recurse into anything other than mods, since
        // mods or tests inside of functions will break things
        if let ast::ItemKind::Mod(mut module) = item.kind {
            let tests = mem::take(&mut self.tests);
            noop_visit_mod(&mut module, self);
            let mut tests = mem::replace(&mut self.tests, tests);

            if !tests.is_empty() {
                let parent =
                    if item.id == ast::DUMMY_NODE_ID { ast::CRATE_NODE_ID } else { item.id };
                // Create an identifier that will hygienically resolve the test
                // case name, even in another module.
                let expn_id = self.cx.ext_cx.resolver.expansion_for_ast_pass(
                    module.inner,
                    AstPass::TestHarness,
                    &[],
                    Some(parent),
                );
                for test in &mut tests {
                    // See the comment on `mk_main` for why we're using
                    // `apply_mark` directly.
                    test.ident.span = test.ident.span.apply_mark(expn_id, Transparency::Opaque);
                }
                self.cx.test_cases.extend(tests);
            }
            item.kind = ast::ItemKind::Mod(module);
        }
        smallvec![P(item)]
    }
}

// Beware, this is duplicated in librustc_passes/entry.rs (with
// `rustc_hir::Item`), so make sure to keep them in sync.
fn entry_point_type(sess: &Session, item: &ast::Item, depth: usize) -> EntryPointType {
    match item.kind {
        ast::ItemKind::Fn(..) => {
            if sess.contains_name(&item.attrs, sym::start) {
                EntryPointType::Start
            } else if sess.contains_name(&item.attrs, sym::main) {
                EntryPointType::MainAttr
            } else if item.ident.name == sym::main {
                if depth == 1 {
                    // This is a top-level function so can be 'main'
                    EntryPointType::MainNamed
                } else {
                    EntryPointType::OtherMain
                }
            } else {
                EntryPointType::None
            }
        }
        _ => EntryPointType::None,
    }
}
/// A folder used to remove any entry points (like fn main) because the harness
/// generator will provide its own
struct EntryPointCleaner<'a> {
    // Current depth in the ast
    sess: &'a Session,
    depth: usize,
    def_site: Span,
}

impl<'a> MutVisitor for EntryPointCleaner<'a> {
    fn flat_map_item(&mut self, i: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        self.depth += 1;
        let item = noop_flat_map_item(i, self).expect_one("noop did something");
        self.depth -= 1;

        // Remove any #[main] or #[start] from the AST so it doesn't
        // clash with the one we're going to add, but mark it as
        // #[allow(dead_code)] to avoid printing warnings.
        let item = match entry_point_type(self.sess, &item, self.depth) {
            EntryPointType::MainNamed | EntryPointType::MainAttr | EntryPointType::Start => item
                .map(|ast::Item { id, ident, attrs, kind, vis, span, tokens }| {
                    let allow_ident = Ident::new(sym::allow, self.def_site);
                    let dc_nested =
                        attr::mk_nested_word_item(Ident::new(sym::dead_code, self.def_site));
                    let allow_dead_code_item = attr::mk_list_item(allow_ident, vec![dc_nested]);
                    let allow_dead_code = attr::mk_attr_outer(allow_dead_code_item);
                    let attrs = attrs
                        .into_iter()
                        .filter(|attr| {
                            !self.sess.check_name(attr, sym::main)
                                && !self.sess.check_name(attr, sym::start)
                        })
                        .chain(iter::once(allow_dead_code))
                        .collect();

                    ast::Item { id, ident, attrs, kind, vis, span, tokens }
                }),
            EntryPointType::None | EntryPointType::OtherMain => item,
        };

        smallvec![item]
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
    let mut econfig = ExpansionConfig::default("test".to_string());
    econfig.features = Some(features);

    let ext_cx = ExtCtxt::new(sess, econfig, resolver, None);

    let expn_id = ext_cx.resolver.expansion_for_ast_pass(
        DUMMY_SP,
        AstPass::TestHarness,
        &[sym::main, sym::test, sym::rustc_attrs],
        None,
    );
    let def_site = DUMMY_SP.with_def_site_ctxt(expn_id);

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
/// ```
/// #[main]
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
/// generated  in `TestHarnessGenerator::flat_map_item`. When resolving this
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
    let test_id = Ident::new(sym::test, sp);

    let runner_name = match cx.panic_strategy {
        PanicStrategy::Unwind => "test_main_static",
        PanicStrategy::Abort => "test_main_static_abort",
    };

    // test::test_main_static(...)
    let mut test_runner = cx
        .test_runner
        .clone()
        .unwrap_or(ecx.path(sp, vec![test_id, Ident::from_str_and_span(runner_name, sp)]));

    test_runner.span = sp;

    let test_main_path_expr = ecx.expr_path(test_runner);
    let call_test_main = ecx.expr_call(sp, test_main_path_expr, vec![mk_tests_slice(cx, sp)]);
    let call_test_main = ecx.stmt_expr(call_test_main);

    // extern crate test
    let test_extern_stmt =
        ecx.stmt_item(sp, ecx.item(sp, test_id, vec![], ast::ItemKind::ExternCrate(None)));

    // #[main]
    let main_meta = ecx.meta_word(sp, sym::main);
    let main_attr = ecx.attribute(main_meta);

    // pub fn main() { ... }
    let main_ret_ty = ecx.ty(sp, ast::TyKind::Tup(vec![]));

    // If no test runner is provided we need to import the test crate
    let main_body = if cx.test_runner.is_none() {
        ecx.block(sp, vec![test_extern_stmt, call_test_main])
    } else {
        ecx.block(sp, vec![call_test_main])
    };

    let decl = ecx.fn_decl(vec![], ast::FnRetTy::Ty(main_ret_ty));
    let sig = ast::FnSig { decl, header: ast::FnHeader::default(), span: sp };
    let def = ast::Defaultness::Final;
    let main = ast::ItemKind::Fn(def, sig, ast::Generics::default(), Some(main_body));

    // Honor the reexport_test_harness_main attribute
    let main_id = match cx.reexport_test_harness_main {
        Some(sym) => Ident::new(sym, sp.with_ctxt(SyntaxContext::root())),
        None => Ident::new(sym::main, sp),
    };

    let main = P(ast::Item {
        ident: main_id,
        attrs: vec![main_attr],
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

    ecx.expr_vec_slice(
        sp,
        cx.test_cases
            .iter()
            .map(|test| {
                ecx.expr_addr_of(test.span, ecx.expr_path(ecx.path(test.span, vec![test.ident])))
            })
            .collect(),
    )
}

fn is_test_case(sess: &Session, i: &ast::Item) -> bool {
    sess.contains_name(&i.attrs, sym::rustc_test_marker)
}

fn get_test_runner(
    sess: &Session,
    sd: &rustc_errors::Handler,
    krate: &ast::Crate,
) -> Option<ast::Path> {
    let test_attr = sess.find_by_name(&krate.attrs, sym::test_runner)?;
    let meta_list = test_attr.meta_item_list()?;
    let span = test_attr.span;
    match &*meta_list {
        [single] => match single.meta_item() {
            Some(meta_item) if meta_item.is_word() => return Some(meta_item.path.clone()),
            _ => sd.struct_span_err(span, "`test_runner` argument must be a path").emit(),
        },
        _ => sd.struct_span_err(span, "`#![test_runner(..)]` accepts exactly 1 argument").emit(),
    }
    None
}
