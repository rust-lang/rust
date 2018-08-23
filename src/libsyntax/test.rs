// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Code that generates a test runner to run all the tests in a crate

#![allow(dead_code)]
#![allow(unused_imports)]

use self::HasTestSignature::*;

use std::iter;
use std::slice;
use std::mem;
use std::vec;
use attr::{self, HasAttrs};
use syntax_pos::{self, DUMMY_SP, NO_EXPANSION, Span, SourceFile, BytePos};

use source_map::{self, SourceMap, ExpnInfo, MacroAttribute, dummy_spanned};
use errors;
use config;
use entry::{self, EntryPointType};
use ext::base::{ExtCtxt, Resolver};
use ext::build::AstBuilder;
use ext::expand::ExpansionConfig;
use ext::hygiene::{self, Mark, SyntaxContext};
use fold::Folder;
use feature_gate::Features;
use util::move_map::MoveMap;
use fold;
use parse::{token, ParseSess};
use print::pprust;
use ast::{self, Ident};
use ptr::P;
use OneVector;
use symbol::{self, Symbol, keywords};
use ThinVec;
use rustc_data_structures::small_vec::ExpectOne;

enum ShouldPanic {
    No,
    Yes(Option<Symbol>),
}

struct Test {
    span: Span,
    path: Vec<Ident> ,
    bench: bool,
    ignore: bool,
    should_panic: ShouldPanic,
    allow_fail: bool,
}

struct TestCtxt<'a> {
    span_diagnostic: &'a errors::Handler,
    path: Vec<Ident>,
    ext_cx: ExtCtxt<'a>,
    testfns: Vec<Test>,
    reexport_test_harness_main: Option<Symbol>,
    is_libtest: bool,
    ctxt: SyntaxContext,
    features: &'a Features,

    // top-level re-export submodule, filled out after folding is finished
    toplevel_reexport: Option<Ident>,
}

// Traverse the crate, collecting all the test functions, eliding any
// existing main functions, and synthesizing a main test harness
pub fn modify_for_testing(sess: &ParseSess,
                          resolver: &mut dyn Resolver,
                          should_test: bool,
                          krate: ast::Crate,
                          span_diagnostic: &errors::Handler,
                          features: &Features) -> ast::Crate {
    // Check for #[reexport_test_harness_main = "some_name"] which
    // creates a `use __test::main as some_name;`. This needs to be
    // unconditional, so that the attribute is still marked as used in
    // non-test builds.
    let reexport_test_harness_main =
        attr::first_attr_value_str_by_name(&krate.attrs,
                                           "reexport_test_harness_main");

    if should_test {
        generate_test_harness(sess, resolver, reexport_test_harness_main,
                              krate, span_diagnostic, features)
    } else {
        krate
    }
}

struct TestHarnessGenerator<'a> {
    cx: TestCtxt<'a>,
    tests: Vec<Ident>,

    // submodule name, gensym'd identifier for re-exports
    tested_submods: Vec<(Ident, Ident)>,
}

impl<'a> fold::Folder for TestHarnessGenerator<'a> {
    fn fold_crate(&mut self, c: ast::Crate) -> ast::Crate {
        let mut folded = fold::noop_fold_crate(c, self);

        // Add a special __test module to the crate that will contain code
        // generated for the test harness
        let (mod_, reexport) = mk_test_module(&mut self.cx);
        if let Some(re) = reexport {
            folded.module.items.push(re)
        }
        folded.module.items.push(mod_);
        folded
    }

    fn fold_item(&mut self, i: P<ast::Item>) -> OneVector<P<ast::Item>> {
        let ident = i.ident;
        if ident.name != keywords::Invalid.name() {
            self.cx.path.push(ident);
        }
        debug!("current path: {}", path_name_i(&self.cx.path));

        if is_test_fn(&self.cx, &i) || is_bench_fn(&self.cx, &i) {
            match i.node {
                ast::ItemKind::Fn(_, header, _, _) => {
                    if header.unsafety == ast::Unsafety::Unsafe {
                        let diag = self.cx.span_diagnostic;
                        diag.span_fatal(
                            i.span,
                            "unsafe functions cannot be used for tests"
                        ).raise();
                    }
                    if header.asyncness.is_async() {
                        let diag = self.cx.span_diagnostic;
                        diag.span_fatal(
                            i.span,
                            "async functions cannot be used for tests"
                        ).raise();
                    }
                }
                _ => {},
            }

            debug!("this is a test function");
            let test = Test {
                span: i.span,
                path: self.cx.path.clone(),
                bench: is_bench_fn(&self.cx, &i),
                ignore: is_ignored(&i),
                should_panic: should_panic(&i, &self.cx),
                allow_fail: is_allowed_fail(&i),
            };
            self.cx.testfns.push(test);
            self.tests.push(i.ident);
        }

        let mut item = i.into_inner();
        // We don't want to recurse into anything other than mods, since
        // mods or tests inside of functions will break things
        if let ast::ItemKind::Mod(module) = item.node {
            let tests = mem::replace(&mut self.tests, Vec::new());
            let tested_submods = mem::replace(&mut self.tested_submods, Vec::new());
            let mut mod_folded = fold::noop_fold_mod(module, self);
            let tests = mem::replace(&mut self.tests, tests);
            let tested_submods = mem::replace(&mut self.tested_submods, tested_submods);

            if !tests.is_empty() || !tested_submods.is_empty() {
                let (it, sym) = mk_reexport_mod(&mut self.cx, item.id, tests, tested_submods);
                mod_folded.items.push(it);

                if !self.cx.path.is_empty() {
                    self.tested_submods.push((self.cx.path[self.cx.path.len()-1], sym));
                } else {
                    debug!("pushing nothing, sym: {:?}", sym);
                    self.cx.toplevel_reexport = Some(sym);
                }
            }
            item.node = ast::ItemKind::Mod(mod_folded);
        }
        if ident.name != keywords::Invalid.name() {
            self.cx.path.pop();
        }
        smallvec![P(item)]
    }

    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac { mac }
}

struct EntryPointCleaner {
    // Current depth in the ast
    depth: usize,
}

impl fold::Folder for EntryPointCleaner {
    fn fold_item(&mut self, i: P<ast::Item>) -> OneVector<P<ast::Item>> {
        self.depth += 1;
        let folded = fold::noop_fold_item(i, self).expect_one("noop did something");
        self.depth -= 1;

        // Remove any #[main] or #[start] from the AST so it doesn't
        // clash with the one we're going to add, but mark it as
        // #[allow(dead_code)] to avoid printing warnings.
        let folded = match entry::entry_point_type(&folded, self.depth) {
            EntryPointType::MainNamed |
            EntryPointType::MainAttr |
            EntryPointType::Start =>
                folded.map(|ast::Item {id, ident, attrs, node, vis, span, tokens}| {
                    let allow_ident = Ident::from_str("allow");
                    let dc_nested = attr::mk_nested_word_item(Ident::from_str("dead_code"));
                    let allow_dead_code_item = attr::mk_list_item(DUMMY_SP, allow_ident,
                                                                  vec![dc_nested]);
                    let allow_dead_code = attr::mk_attr_outer(DUMMY_SP,
                                                              attr::mk_attr_id(),
                                                              allow_dead_code_item);

                    ast::Item {
                        id,
                        ident,
                        attrs: attrs.into_iter()
                            .filter(|attr| {
                                !attr.check_name("main") && !attr.check_name("start")
                            })
                            .chain(iter::once(allow_dead_code))
                            .collect(),
                        node,
                        vis,
                        span,
                        tokens,
                    }
                }),
            EntryPointType::None |
            EntryPointType::OtherMain => folded,
        };

        smallvec![folded]
    }

    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac { mac }
}

fn mk_reexport_mod(cx: &mut TestCtxt,
                   parent: ast::NodeId,
                   tests: Vec<Ident>,
                   tested_submods: Vec<(Ident, Ident)>)
                   -> (P<ast::Item>, Ident) {
    let super_ = Ident::from_str("super");

    let items = tests.into_iter().map(|r| {
        cx.ext_cx.item_use_simple(DUMMY_SP, dummy_spanned(ast::VisibilityKind::Public),
                                  cx.ext_cx.path(DUMMY_SP, vec![super_, r]))
    }).chain(tested_submods.into_iter().map(|(r, sym)| {
        let path = cx.ext_cx.path(DUMMY_SP, vec![super_, r, sym]);
        cx.ext_cx.item_use_simple_(DUMMY_SP, dummy_spanned(ast::VisibilityKind::Public),
                                   Some(r), path)
    })).collect();

    let reexport_mod = ast::Mod {
        inner: DUMMY_SP,
        items,
    };

    let sym = Ident::with_empty_ctxt(Symbol::gensym("__test_reexports"));
    let parent = if parent == ast::DUMMY_NODE_ID { ast::CRATE_NODE_ID } else { parent };
    cx.ext_cx.current_expansion.mark = cx.ext_cx.resolver.get_module_scope(parent);
    let it = cx.ext_cx.monotonic_expander().fold_item(P(ast::Item {
        ident: sym,
        attrs: Vec::new(),
        id: ast::DUMMY_NODE_ID,
        node: ast::ItemKind::Mod(reexport_mod),
        vis: dummy_spanned(ast::VisibilityKind::Public),
        span: DUMMY_SP,
        tokens: None,
    })).pop().unwrap();

    (it, sym)
}

fn generate_test_harness(sess: &ParseSess,
                         resolver: &mut dyn Resolver,
                         reexport_test_harness_main: Option<Symbol>,
                         krate: ast::Crate,
                         sd: &errors::Handler,
                         features: &Features) -> ast::Crate {
    // Remove the entry points
    let mut cleaner = EntryPointCleaner { depth: 0 };
    let krate = cleaner.fold_crate(krate);

    let mark = Mark::fresh(Mark::root());

    let mut econfig = ExpansionConfig::default("test".to_string());
    econfig.features = Some(features);

    let cx = TestCtxt {
        span_diagnostic: sd,
        ext_cx: ExtCtxt::new(sess, econfig, resolver),
        path: Vec::new(),
        testfns: Vec::new(),
        reexport_test_harness_main,
        // NB: doesn't consider the value of `--crate-name` passed on the command line.
        is_libtest: attr::find_crate_name(&krate.attrs).map(|s| s == "test").unwrap_or(false),
        toplevel_reexport: None,
        ctxt: SyntaxContext::empty().apply_mark(mark),
        features,
    };

    mark.set_expn_info(ExpnInfo {
        call_site: DUMMY_SP,
        def_site: None,
        format: MacroAttribute(Symbol::intern("test")),
        allow_internal_unstable: true,
        allow_internal_unsafe: false,
        local_inner_macros: false,
        edition: hygiene::default_edition(),
    });

    TestHarnessGenerator {
        cx,
        tests: Vec::new(),
        tested_submods: Vec::new(),
    }.fold_crate(krate)
}

/// Craft a span that will be ignored by the stability lint's
/// call to source_map's `is_internal` check.
/// The expanded code calls some unstable functions in the test crate.
fn ignored_span(cx: &TestCtxt, sp: Span) -> Span {
    sp.with_ctxt(cx.ctxt)
}

enum HasTestSignature {
    Yes,
    No(BadTestSignature),
}

#[derive(PartialEq)]
enum BadTestSignature {
    NotEvenAFunction,
    WrongTypeSignature,
    NoArgumentsAllowed,
    ShouldPanicOnlyWithNoArgs,
}

fn is_test_fn(cx: &TestCtxt, i: &ast::Item) -> bool {
    let has_test_attr = attr::contains_name(&i.attrs, "test");

    fn has_test_signature(_cx: &TestCtxt, i: &ast::Item) -> HasTestSignature {
        let has_should_panic_attr = attr::contains_name(&i.attrs, "should_panic");
        match i.node {
            ast::ItemKind::Fn(ref decl, _, ref generics, _) => {
                // If the termination trait is active, the compiler will check that the output
                // type implements the `Termination` trait as `libtest` enforces that.
                let has_output = match decl.output {
                    ast::FunctionRetTy::Default(..) => false,
                    ast::FunctionRetTy::Ty(ref t) if t.node.is_unit() => false,
                    _ => true
                };

                if !decl.inputs.is_empty() {
                    return No(BadTestSignature::NoArgumentsAllowed);
                }

                match (has_output, has_should_panic_attr) {
                    (true, true) => No(BadTestSignature::ShouldPanicOnlyWithNoArgs),
                    (true, false) => if !generics.params.is_empty() {
                        No(BadTestSignature::WrongTypeSignature)
                    } else {
                        Yes
                    },
                    (false, _) => Yes
                }
            }
            _ => No(BadTestSignature::NotEvenAFunction),
        }
    }

    let has_test_signature = if has_test_attr {
        let diag = cx.span_diagnostic;
        match has_test_signature(cx, i) {
            Yes => true,
            No(cause) => {
                match cause {
                    BadTestSignature::NotEvenAFunction =>
                        diag.span_err(i.span, "only functions may be used as tests"),
                    BadTestSignature::WrongTypeSignature =>
                        diag.span_err(i.span,
                                      "functions used as tests must have signature fn() -> ()"),
                    BadTestSignature::NoArgumentsAllowed =>
                        diag.span_err(i.span, "functions used as tests can not have any arguments"),
                    BadTestSignature::ShouldPanicOnlyWithNoArgs =>
                        diag.span_err(i.span, "functions using `#[should_panic]` must return `()`"),
                }
                false
            }
        }
    } else {
        false
    };

    has_test_attr && has_test_signature
}

fn is_bench_fn(cx: &TestCtxt, i: &ast::Item) -> bool {
    let has_bench_attr = attr::contains_name(&i.attrs, "bench");

    fn has_bench_signature(_cx: &TestCtxt, i: &ast::Item) -> bool {
        match i.node {
            ast::ItemKind::Fn(ref decl, _, _, _) => {
                // NB: inadequate check, but we're running
                // well before resolve, can't get too deep.
                decl.inputs.len() == 1
            }
            _ => false
        }
    }

    let has_bench_signature = has_bench_signature(cx, i);

    if has_bench_attr && !has_bench_signature {
        let diag = cx.span_diagnostic;

        diag.span_err(i.span, "functions used as benches must have signature \
                                   `fn(&mut Bencher) -> impl Termination`");
    }

    has_bench_attr && has_bench_signature
}

fn is_ignored(i: &ast::Item) -> bool {
    attr::contains_name(&i.attrs, "ignore")
}

fn is_allowed_fail(i: &ast::Item) -> bool {
    attr::contains_name(&i.attrs, "allow_fail")
}

fn should_panic(i: &ast::Item, cx: &TestCtxt) -> ShouldPanic {
    match attr::find_by_name(&i.attrs, "should_panic") {
        Some(attr) => {
            let sd = cx.span_diagnostic;
            if attr.is_value_str() {
                sd.struct_span_warn(
                    attr.span(),
                    "attribute must be of the form: \
                     `#[should_panic]` or \
                     `#[should_panic(expected = \"error message\")]`"
                ).note("Errors in this attribute were erroneously allowed \
                        and will become a hard error in a future release.")
                .emit();
                return ShouldPanic::Yes(None);
            }
            match attr.meta_item_list() {
                // Handle #[should_panic]
                None => ShouldPanic::Yes(None),
                // Handle #[should_panic(expected = "foo")]
                Some(list) => {
                    let msg = list.iter()
                        .find(|mi| mi.check_name("expected"))
                        .and_then(|mi| mi.meta_item())
                        .and_then(|mi| mi.value_str());
                    if list.len() != 1 || msg.is_none() {
                        sd.struct_span_warn(
                            attr.span(),
                            "argument must be of the form: \
                             `expected = \"error message\"`"
                        ).note("Errors in this attribute were erroneously \
                                allowed and will become a hard error in a \
                                future release.").emit();
                        ShouldPanic::Yes(None)
                    } else {
                        ShouldPanic::Yes(msg)
                    }
                },
            }
        }
        None => ShouldPanic::No,
    }
}

/*

We're going to be building a module that looks more or less like:

mod __test {
  extern crate test (name = "test", vers = "...");
  fn main() {
    test::test_main_static(&::os::args()[], tests, test::Options::new())
  }

  static tests : &'static [test::TestDescAndFn] = &[
    ... the list of tests in the crate ...
  ];
}

*/

fn mk_std(cx: &TestCtxt) -> P<ast::Item> {
    let id_test = Ident::from_str("test");
    let sp = ignored_span(cx, DUMMY_SP);
    let (vi, vis, ident) = if cx.is_libtest {
        (ast::ItemKind::Use(P(ast::UseTree {
            span: DUMMY_SP,
            prefix: path_node(vec![id_test]),
            kind: ast::UseTreeKind::Simple(None, ast::DUMMY_NODE_ID, ast::DUMMY_NODE_ID),
        })),
         ast::VisibilityKind::Public, keywords::Invalid.ident())
    } else {
        (ast::ItemKind::ExternCrate(None), ast::VisibilityKind::Inherited, id_test)
    };
    P(ast::Item {
        id: ast::DUMMY_NODE_ID,
        ident,
        node: vi,
        attrs: vec![],
        vis: dummy_spanned(vis),
        span: sp,
        tokens: None,
    })
}

fn mk_main(cx: &mut TestCtxt) -> P<ast::Item> {
    // Writing this out by hand with 'ignored_span':
    //        pub fn main() {
    //            #![main]
    //            use std::slice::AsSlice;
    //            test::test_main_static(::std::os::args().as_slice(), TESTS, test::Options::new());
    //        }

    let sp = ignored_span(cx, DUMMY_SP);
    let ecx = &cx.ext_cx;

    // test::test_main_static
    let test_main_path =
        ecx.path(sp, vec![Ident::from_str("test"), Ident::from_str("test_main_static")]);

    // test::test_main_static(...)
    let test_main_path_expr = ecx.expr_path(test_main_path);
    let tests_ident_expr = ecx.expr_ident(sp, Ident::from_str("TESTS"));
    let call_test_main = ecx.expr_call(sp, test_main_path_expr,
                                       vec![tests_ident_expr]);
    let call_test_main = ecx.stmt_expr(call_test_main);
    // #![main]
    let main_meta = ecx.meta_word(sp, Symbol::intern("main"));
    let main_attr = ecx.attribute(sp, main_meta);
    // pub fn main() { ... }
    let main_ret_ty = ecx.ty(sp, ast::TyKind::Tup(vec![]));
    let main_body = ecx.block(sp, vec![call_test_main]);
    let main = ast::ItemKind::Fn(ecx.fn_decl(vec![], ast::FunctionRetTy::Ty(main_ret_ty)),
                           ast::FnHeader::default(),
                           ast::Generics::default(),
                           main_body);
    P(ast::Item {
        ident: Ident::from_str("main"),
        attrs: vec![main_attr],
        id: ast::DUMMY_NODE_ID,
        node: main,
        vis: dummy_spanned(ast::VisibilityKind::Public),
        span: sp,
        tokens: None,
    })
}

fn mk_test_module(cx: &mut TestCtxt) -> (P<ast::Item>, Option<P<ast::Item>>) {
    // Link to test crate
    let import = mk_std(cx);

    // A constant vector of test descriptors.
    let tests = mk_tests(cx);

    // The synthesized main function which will call the console test runner
    // with our list of tests
    let mainfn = mk_main(cx);

    let testmod = ast::Mod {
        inner: DUMMY_SP,
        items: vec![import, mainfn, tests],
    };
    let item_ = ast::ItemKind::Mod(testmod);
    let mod_ident = Ident::with_empty_ctxt(Symbol::gensym("__test"));

    let mut expander = cx.ext_cx.monotonic_expander();
    let item = expander.fold_item(P(ast::Item {
        id: ast::DUMMY_NODE_ID,
        ident: mod_ident,
        attrs: vec![],
        node: item_,
        vis: dummy_spanned(ast::VisibilityKind::Public),
        span: DUMMY_SP,
        tokens: None,
    })).pop().unwrap();
    let reexport = cx.reexport_test_harness_main.map(|s| {
        // building `use __test::main as <ident>;`
        let rename = Ident::with_empty_ctxt(s);

        let use_path = ast::UseTree {
            span: DUMMY_SP,
            prefix: path_node(vec![mod_ident, Ident::from_str("main")]),
            kind: ast::UseTreeKind::Simple(Some(rename), ast::DUMMY_NODE_ID, ast::DUMMY_NODE_ID),
        };

        expander.fold_item(P(ast::Item {
            id: ast::DUMMY_NODE_ID,
            ident: keywords::Invalid.ident(),
            attrs: vec![],
            node: ast::ItemKind::Use(P(use_path)),
            vis: dummy_spanned(ast::VisibilityKind::Inherited),
            span: DUMMY_SP,
            tokens: None,
        })).pop().unwrap()
    });

    debug!("Synthetic test module:\n{}\n", pprust::item_to_string(&item));

    (item, reexport)
}

fn nospan<T>(t: T) -> source_map::Spanned<T> {
    source_map::Spanned { node: t, span: DUMMY_SP }
}

fn path_node(ids: Vec<Ident>) -> ast::Path {
    ast::Path {
        span: DUMMY_SP,
        segments: ids.into_iter().map(|id| ast::PathSegment::from_ident(id)).collect(),
    }
}

fn path_name_i(idents: &[Ident]) -> String {
    let mut path_name = "".to_string();
    let mut idents_iter = idents.iter().peekable();
    while let Some(ident) = idents_iter.next() {
        path_name.push_str(&ident.as_str());
        if idents_iter.peek().is_some() {
            path_name.push_str("::")
        }
    }
    path_name
}

fn mk_tests(cx: &TestCtxt) -> P<ast::Item> {
    // The vector of test_descs for this crate
    let test_descs = mk_test_descs(cx);

    // FIXME #15962: should be using quote_item, but that stringifies
    // __test_reexports, causing it to be reinterned, losing the
    // gensym information.
    let sp = ignored_span(cx, DUMMY_SP);
    let ecx = &cx.ext_cx;
    let struct_type = ecx.ty_path(ecx.path(sp, vec![ecx.ident_of("self"),
                                                    ecx.ident_of("test"),
                                                    ecx.ident_of("TestDescAndFn")]));
    let static_lt = ecx.lifetime(sp, keywords::StaticLifetime.ident());
    // &'static [self::test::TestDescAndFn]
    let static_type = ecx.ty_rptr(sp,
                                  ecx.ty(sp, ast::TyKind::Slice(struct_type)),
                                  Some(static_lt),
                                  ast::Mutability::Immutable);
    // static TESTS: $static_type = &[...];
    ecx.item_const(sp,
                   ecx.ident_of("TESTS"),
                   static_type,
                   test_descs)
}

fn mk_test_descs(cx: &TestCtxt) -> P<ast::Expr> {
    debug!("building test vector from {} tests", cx.testfns.len());

    P(ast::Expr {
        id: ast::DUMMY_NODE_ID,
        node: ast::ExprKind::AddrOf(ast::Mutability::Immutable,
            P(ast::Expr {
                id: ast::DUMMY_NODE_ID,
                node: ast::ExprKind::Array(cx.testfns.iter().map(|test| {
                    mk_test_desc_and_fn_rec(cx, test)
                }).collect()),
                span: DUMMY_SP,
                attrs: ThinVec::new(),
            })),
        span: DUMMY_SP,
        attrs: ThinVec::new(),
    })
}

fn mk_test_desc_and_fn_rec(cx: &TestCtxt, test: &Test) -> P<ast::Expr> {
    // FIXME #15962: should be using quote_expr, but that stringifies
    // __test_reexports, causing it to be reinterned, losing the
    // gensym information.

    let span = ignored_span(cx, test.span);
    let ecx = &cx.ext_cx;
    let self_id = ecx.ident_of("self");
    let test_id = ecx.ident_of("test");

    // creates self::test::$name
    let test_path = |name| {
        ecx.path(span, vec![self_id, test_id, ecx.ident_of(name)])
    };
    // creates $name: $expr
    let field = |name, expr| ecx.field_imm(span, ecx.ident_of(name), expr);

    // path to the #[test] function: "foo::bar::baz"
    let path_string = path_name_i(&test.path[..]);

    debug!("encoding {}", path_string);

    let name_expr = ecx.expr_str(span, Symbol::intern(&path_string));

    // self::test::StaticTestName($name_expr)
    let name_expr = ecx.expr_call(span,
                                  ecx.expr_path(test_path("StaticTestName")),
                                  vec![name_expr]);

    let ignore_expr = ecx.expr_bool(span, test.ignore);
    let should_panic_path = |name| {
        ecx.path(span, vec![self_id, test_id, ecx.ident_of("ShouldPanic"), ecx.ident_of(name)])
    };
    let fail_expr = match test.should_panic {
        ShouldPanic::No => ecx.expr_path(should_panic_path("No")),
        ShouldPanic::Yes(msg) => {
            match msg {
                Some(msg) => {
                    let msg = ecx.expr_str(span, msg);
                    let path = should_panic_path("YesWithMessage");
                    ecx.expr_call(span, ecx.expr_path(path), vec![msg])
                }
                None => ecx.expr_path(should_panic_path("Yes")),
            }
        }
    };
    let allow_fail_expr = ecx.expr_bool(span, test.allow_fail);

    // self::test::TestDesc { ... }
    let desc_expr = ecx.expr_struct(
        span,
        test_path("TestDesc"),
        vec![field("name", name_expr),
             field("ignore", ignore_expr),
             field("should_panic", fail_expr),
             field("allow_fail", allow_fail_expr)]);

    let mut visible_path = vec![];
    if cx.features.extern_absolute_paths {
        visible_path.push(keywords::Crate.ident());
    }
    match cx.toplevel_reexport {
        Some(id) => visible_path.push(id),
        None => {
            let diag = cx.span_diagnostic;
            diag.bug("expected to find top-level re-export name, but found None");
        }
    };
    visible_path.extend_from_slice(&test.path[..]);

    // Rather than directly give the test function to the test
    // harness, we create a wrapper like one of the following:
    //
    //     || test::assert_test_result(real_function()) // for test
    //     |b| test::assert_test_result(real_function(b)) // for bench
    //
    // this will coerce into a fn pointer that is specialized to the
    // actual return type of `real_function` (Typically `()`, but not always).
    let fn_expr = {
        // construct `real_function()` (this will be inserted into the overall expr)
        let real_function_expr = ecx.expr_path(ecx.path_global(span, visible_path));
        // construct path `test::assert_test_result`
        let assert_test_result = test_path("assert_test_result");
        if test.bench {
            // construct `|b| {..}`
            let b_ident = Ident::with_empty_ctxt(Symbol::gensym("b"));
            let b_expr = ecx.expr_ident(span, b_ident);
            ecx.lambda(
                span,
                vec![b_ident],
                // construct `assert_test_result(..)`
                ecx.expr_call(
                    span,
                    ecx.expr_path(assert_test_result),
                    vec![
                        // construct `real_function(b)`
                        ecx.expr_call(
                            span,
                            real_function_expr,
                            vec![b_expr],
                        )
                    ],
                ),
            )
        } else {
            // construct `|| {..}`
            ecx.lambda(
                span,
                vec![],
                // construct `assert_test_result(..)`
                ecx.expr_call(
                    span,
                    ecx.expr_path(assert_test_result),
                    vec![
                        // construct `real_function()`
                        ecx.expr_call(
                            span,
                            real_function_expr,
                            vec![],
                        )
                    ],
                ),
            )
        }
    };

    let variant_name = if test.bench { "StaticBenchFn" } else { "StaticTestFn" };

    // self::test::$variant_name($fn_expr)
    let testfn_expr = ecx.expr_call(span, ecx.expr_path(test_path(variant_name)), vec![fn_expr]);

    // self::test::TestDescAndFn { ... }
    ecx.expr_struct(span,
                    test_path("TestDescAndFn"),
                    vec![field("desc", desc_expr),
                         field("testfn", testfn_expr)])
}
