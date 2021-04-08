// Code that generates a test runner to run all the tests in a crate

use rustc_ast as ast;
use rustc_ast::mut_visit::{ExpectOne, *};
use rustc_ast::ptr::P;
use rustc_ast::{attr, ModKind};
use rustc_expand::base::{ExtCtxt, ResolverExpand};
use rustc_expand::expand::{AstFragment, ExpansionConfig};
use rustc_feature::Features;
use rustc_session::config::CrateType;
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
    let platform_panic_strategy = sess.target.panic_strategy;

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

        ensure_exist_rustc_main_attr(&mut self.cx, &mut c.attrs);
        // Create an entry function to run our tests
        mk_entry_fn(&mut self.cx, c);
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
        if let ast::ItemKind::Mod(..) = item.kind {
            let tests = mem::take(&mut self.tests);
            noop_visit_item_kind(&mut item.kind, self);
            let mut tests = mem::replace(&mut self.tests, tests);

            if !tests.is_empty() {
                let parent =
                    if item.id == ast::DUMMY_NODE_ID { ast::CRATE_NODE_ID } else { item.id };
                // Create an identifier that will hygienically resolve the test
                // case name, even in another module.
                let inner_span = match item.kind {
                    ast::ItemKind::Mod(_, ModKind::Loaded(.., span)) => span,
                    _ => unreachable!(),
                };
                let expn_id = self.cx.ext_cx.resolver.expansion_for_ast_pass(
                    inner_span,
                    AstPass::TestHarness,
                    &[],
                    Some(parent),
                );
                for test in &mut tests {
                    // See the comment on `mk_entry_fn` for why we're using
                    // `apply_mark` directly.
                    test.ident.span = test.ident.span.apply_mark(expn_id, Transparency::Opaque);
                }
                self.cx.test_cases.extend(tests);
            }
        }
        smallvec![P(item)]
    }
}

/// Remove any #[start] from the AST so it doesn't
/// clash with the one we're going to add, but mark it as
/// #[allow(dead_code)] to avoid printing warnings.
fn strip_start_attr(sess: &Session, def_site: Span, item: P<ast::Item>) -> P<ast::Item> {
    if !matches!(item.kind, ast::ItemKind::Fn(..)) {
        return item;
    }
    if !sess.contains_name(&item.attrs, sym::start) {
        return item;
    }

    item.map(|item| {
        let ast::Item { id, ident, attrs, kind, vis, span, tokens } = item;

        let allow_ident = Ident::new(sym::allow, def_site);
        let dc_nested = attr::mk_nested_word_item(Ident::new(sym::dead_code, def_site));
        let allow_dead_code_item = attr::mk_list_item(allow_ident, vec![dc_nested]);
        let allow_dead_code = attr::mk_attr_outer(allow_dead_code_item);
        let attrs = attrs
            .into_iter()
            .filter(|attr| !sess.check_name(attr, sym::start))
            .chain(iter::once(allow_dead_code))
            .collect();

        ast::Item { id, ident, attrs, kind, vis, span, tokens }
    })
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
        smallvec![strip_start_attr(self.sess, self.def_site, item)]
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
        &[sym::rustc_main, sym::test, sym::rustc_attrs],
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
/// ```rust,ignore (not real code)
/// #![rustc_main(crate::...::main)]
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
fn mk_entry_fn(cx: &mut TestCtxt<'_>, c: &mut ast::Crate) {
    let sp = cx.def_site;
    let ecx = &mut cx.ext_cx;
    let test_id = Ident::new(sym::test, sp);

    let runner_name = match cx.panic_strategy {
        PanicStrategy::Unwind => "test_main_static",
        PanicStrategy::Abort => "test_main_static_abort",
    };

    // test::test_main_static(...)
    let mut test_runner = cx
        .test_runner
        .clone()
        .unwrap_or_else(|| ecx.path(sp, vec![test_id, Ident::from_str_and_span(runner_name, sp)]));

    test_runner.span = sp;

    let test_main_path_expr = ecx.expr_path(test_runner);
    let call_test_main =
        ecx.expr_call(sp, test_main_path_expr, vec![mk_tests_slice(ecx, &cx.test_cases, sp)]);
    let call_test_main = ecx.stmt_expr(call_test_main);

    // extern crate test
    let test_extern_stmt =
        ecx.stmt_item(sp, ecx.item(sp, test_id, vec![], ast::ItemKind::ExternCrate(None)));

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
    let main =
        ast::ItemKind::Fn(box ast::FnKind(def, sig, ast::Generics::default(), Some(main_body)));

    // Honor the reexport_test_harness_main attribute
    let main_id = match cx.reexport_test_harness_main {
        Some(sym) => Ident::new(sym, sp.with_ctxt(SyntaxContext::root())),
        None => Ident::new(sym::main, sp),
    };

    let main = P(ast::Item {
        ident: main_id,
        attrs: vec![],
        id: ast::DUMMY_NODE_ID,
        kind: main,
        vis: ast::Visibility { span: sp, kind: ast::VisibilityKind::Public, tokens: None },
        span: sp,
        tokens: None,
    });

    // Integrate the new item into existing module structures.
    let main = AstFragment::Items(smallvec![main]);
    let main = ecx.monotonic_expander().fully_expand_fragment(main).make_items().pop().unwrap();

    // #[rustc_main] attr
    let main_id_nested_meta = ast::attr::mk_nested_word_item(main_id);
    let rustc_main_meta = ecx.meta_list(sp, sym::rustc_main, vec![main_id_nested_meta]);
    let rustc_main_attr = ecx.attribute(rustc_main_meta);
    c.attrs.push(rustc_main_attr);
    c.items.push(main);
}

fn ensure_exist_rustc_main_attr(cx: &mut TestCtxt<'_>, attrs: &mut Vec<ast::Attribute>) {
    let sp = cx.def_site;
    let ecx = &cx.ext_cx;

    if ecx.sess.contains_name(attrs, sym::rustc_main) {
        return;
    }

    let any_exe = ecx.sess.crate_types().iter().any(|ty| *ty == CrateType::Executable);
    if !any_exe {
        return;
    }

    if ecx.sess.contains_name(attrs, sym::no_main) {
        return;
    }

    let crate_main_nested_meta = ast::attr::mk_nested_word_item(Ident::new(sym::main, DUMMY_SP));
    let ignore_nested_meta = ast::attr::mk_nested_word_item(Ident::new(sym::ignore, DUMMY_SP));
    let rustc_main_meta =
        ecx.meta_list(sp, sym::rustc_main, vec![crate_main_nested_meta, ignore_nested_meta]);
    let rustc_main_attr = ecx.attribute(rustc_main_meta);
    attrs.push(rustc_main_attr);
}

/// Creates a slice containing every test like so:
/// &[&test1, &test2]
fn mk_tests_slice(ecx: &ExtCtxt<'_>, test_cases: &Vec<Test>, sp: Span) -> P<ast::Expr> {
    debug!("building test vector from {} tests", test_cases.len());
    ecx.expr_vec_slice(
        sp,
        test_cases
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
