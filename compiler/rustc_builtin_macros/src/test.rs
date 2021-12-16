/// The expansion from a test function to the appropriate test struct for libtest
/// Ideally, this code would be in libtest but for efficiency and error messages it lives here.
use crate::util::{check_builtin_macro_attribute, warn_on_duplicate_attribute};

use rustc_ast as ast;
use rustc_ast::attr;
use rustc_ast::ptr::P;
use rustc_ast_pretty::pprust;
use rustc_expand::base::*;
use rustc_session::Session;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::Span;

use std::iter;

// #[test_case] is used by custom test authors to mark tests
// When building for test, it needs to make the item public and gensym the name
// Otherwise, we'll omit the item. This behavior means that any item annotated
// with #[test_case] is never addressable.
//
// We mark item with an inert attribute "rustc_test_marker" which the test generation
// logic will pick up on.
pub fn expand_test_case(
    ecx: &mut ExtCtxt<'_>,
    attr_sp: Span,
    meta_item: &ast::MetaItem,
    anno_item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::test_case);
    warn_on_duplicate_attribute(&ecx, &anno_item, sym::test_case);

    if !ecx.ecfg.should_test {
        return vec![];
    }

    let sp = ecx.with_def_site_ctxt(attr_sp);
    let mut item = anno_item.expect_item();
    item = item.map(|mut item| {
        item.vis = ast::Visibility {
            span: item.vis.span,
            kind: ast::VisibilityKind::Public,
            tokens: None,
        };
        item.ident.span = item.ident.span.with_ctxt(sp.ctxt());
        item.attrs.push(ecx.attribute(ecx.meta_word(sp, sym::rustc_test_marker)));
        item
    });

    return vec![Annotatable::Item(item)];
}

pub fn expand_test(
    cx: &mut ExtCtxt<'_>,
    attr_sp: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(cx, meta_item, sym::test);
    warn_on_duplicate_attribute(&cx, &item, sym::test);
    expand_test_or_bench(cx, attr_sp, item, false)
}

pub fn expand_bench(
    cx: &mut ExtCtxt<'_>,
    attr_sp: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(cx, meta_item, sym::bench);
    warn_on_duplicate_attribute(&cx, &item, sym::bench);
    expand_test_or_bench(cx, attr_sp, item, true)
}

pub fn expand_test_or_bench(
    cx: &mut ExtCtxt<'_>,
    attr_sp: Span,
    item: Annotatable,
    is_bench: bool,
) -> Vec<Annotatable> {
    // If we're not in test configuration, remove the annotated item
    if !cx.ecfg.should_test {
        return vec![];
    }

    let (item, is_stmt) = match item {
        Annotatable::Item(i) => (i, false),
        Annotatable::Stmt(stmt) if matches!(stmt.kind, ast::StmtKind::Item(_)) => {
            // FIXME: Use an 'if let' guard once they are implemented
            if let ast::StmtKind::Item(i) = stmt.into_inner().kind {
                (i, true)
            } else {
                unreachable!()
            }
        }
        other => {
            cx.struct_span_err(
                other.span(),
                "`#[test]` attribute is only allowed on non associated functions",
            )
            .emit();
            return vec![other];
        }
    };

    if let ast::ItemKind::MacCall(_) = item.kind {
        cx.sess.parse_sess.span_diagnostic.span_warn(
            item.span,
            "`#[test]` attribute should not be used on macros. Use `#[cfg(test)]` instead.",
        );
        return vec![Annotatable::Item(item)];
    }

    // has_*_signature will report any errors in the type so compilation
    // will fail. We shouldn't try to expand in this case because the errors
    // would be spurious.
    if (!is_bench && !has_test_signature(cx, &item))
        || (is_bench && !has_bench_signature(cx, &item))
    {
        return vec![Annotatable::Item(item)];
    }

    let (sp, attr_sp) = (cx.with_def_site_ctxt(item.span), cx.with_def_site_ctxt(attr_sp));

    let test_id = Ident::new(sym::test, attr_sp);

    // creates test::$name
    let test_path = |name| cx.path(sp, vec![test_id, Ident::from_str_and_span(name, sp)]);

    // creates test::ShouldPanic::$name
    let should_panic_path = |name| {
        cx.path(
            sp,
            vec![
                test_id,
                Ident::from_str_and_span("ShouldPanic", sp),
                Ident::from_str_and_span(name, sp),
            ],
        )
    };

    // creates test::TestType::$name
    let test_type_path = |name| {
        cx.path(
            sp,
            vec![
                test_id,
                Ident::from_str_and_span("TestType", sp),
                Ident::from_str_and_span(name, sp),
            ],
        )
    };

    // creates $name: $expr
    let field = |name, expr| cx.field_imm(sp, Ident::from_str_and_span(name, sp), expr);

    let test_fn = if is_bench {
        // A simple ident for a lambda
        let b = Ident::from_str_and_span("b", attr_sp);

        cx.expr_call(
            sp,
            cx.expr_path(test_path("StaticBenchFn")),
            vec![
                // |b| self::test::assert_test_result(
                cx.lambda1(
                    sp,
                    cx.expr_call(
                        sp,
                        cx.expr_path(test_path("assert_test_result")),
                        vec![
                            // super::$test_fn(b)
                            cx.expr_call(
                                sp,
                                cx.expr_path(cx.path(sp, vec![item.ident])),
                                vec![cx.expr_ident(sp, b)],
                            ),
                        ],
                    ),
                    b,
                ), // )
            ],
        )
    } else {
        cx.expr_call(
            sp,
            cx.expr_path(test_path("StaticTestFn")),
            vec![
                // || {
                cx.lambda0(
                    sp,
                    // test::assert_test_result(
                    cx.expr_call(
                        sp,
                        cx.expr_path(test_path("assert_test_result")),
                        vec![
                            // $test_fn()
                            cx.expr_call(sp, cx.expr_path(cx.path(sp, vec![item.ident])), vec![]), // )
                        ],
                    ), // }
                ), // )
            ],
        )
    };

    let mut test_const = cx.item(
        sp,
        Ident::new(item.ident.name, sp),
        vec![
            // #[cfg(test)]
            cx.attribute(attr::mk_list_item(
                Ident::new(sym::cfg, attr_sp),
                vec![attr::mk_nested_word_item(Ident::new(sym::test, attr_sp))],
            )),
            // #[rustc_test_marker]
            cx.attribute(cx.meta_word(attr_sp, sym::rustc_test_marker)),
        ],
        // const $ident: test::TestDescAndFn =
        ast::ItemKind::Const(
            ast::Defaultness::Final,
            cx.ty(sp, ast::TyKind::Path(None, test_path("TestDescAndFn"))),
            // test::TestDescAndFn {
            Some(
                cx.expr_struct(
                    sp,
                    test_path("TestDescAndFn"),
                    vec![
                        // desc: test::TestDesc {
                        field(
                            "desc",
                            cx.expr_struct(
                                sp,
                                test_path("TestDesc"),
                                vec![
                                    // name: "path::to::test"
                                    field(
                                        "name",
                                        cx.expr_call(
                                            sp,
                                            cx.expr_path(test_path("StaticTestName")),
                                            vec![cx.expr_str(
                                                sp,
                                                Symbol::intern(&item_path(
                                                    // skip the name of the root module
                                                    &cx.current_expansion.module.mod_path[1..],
                                                    &item.ident,
                                                )),
                                            )],
                                        ),
                                    ),
                                    // ignore: true | false
                                    field(
                                        "ignore",
                                        cx.expr_bool(sp, should_ignore(&cx.sess, &item)),
                                    ),
                                    // allow_fail: true | false
                                    field(
                                        "allow_fail",
                                        cx.expr_bool(sp, should_fail(&cx.sess, &item)),
                                    ),
                                    // compile_fail: true | false
                                    field("compile_fail", cx.expr_bool(sp, false)),
                                    // no_run: true | false
                                    field("no_run", cx.expr_bool(sp, false)),
                                    // should_panic: ...
                                    field(
                                        "should_panic",
                                        match should_panic(cx, &item) {
                                            // test::ShouldPanic::No
                                            ShouldPanic::No => {
                                                cx.expr_path(should_panic_path("No"))
                                            }
                                            // test::ShouldPanic::Yes
                                            ShouldPanic::Yes(None) => {
                                                cx.expr_path(should_panic_path("Yes"))
                                            }
                                            // test::ShouldPanic::YesWithMessage("...")
                                            ShouldPanic::Yes(Some(sym)) => cx.expr_call(
                                                sp,
                                                cx.expr_path(should_panic_path("YesWithMessage")),
                                                vec![cx.expr_str(sp, sym)],
                                            ),
                                        },
                                    ),
                                    // test_type: ...
                                    field(
                                        "test_type",
                                        match test_type(cx) {
                                            // test::TestType::UnitTest
                                            TestType::UnitTest => {
                                                cx.expr_path(test_type_path("UnitTest"))
                                            }
                                            // test::TestType::IntegrationTest
                                            TestType::IntegrationTest => {
                                                cx.expr_path(test_type_path("IntegrationTest"))
                                            }
                                            // test::TestPath::Unknown
                                            TestType::Unknown => {
                                                cx.expr_path(test_type_path("Unknown"))
                                            }
                                        },
                                    ),
                                    // },
                                ],
                            ),
                        ),
                        // testfn: test::StaticTestFn(...) | test::StaticBenchFn(...)
                        field("testfn", test_fn), // }
                    ],
                ), // }
            ),
        ),
    );
    test_const = test_const.map(|mut tc| {
        tc.vis.kind = ast::VisibilityKind::Public;
        tc
    });

    // extern crate test
    let test_extern = cx.item(sp, test_id, vec![], ast::ItemKind::ExternCrate(None));

    tracing::debug!("synthetic test item:\n{}\n", pprust::item_to_string(&test_const));

    if is_stmt {
        vec![
            // Access to libtest under a hygienic name
            Annotatable::Stmt(P(cx.stmt_item(sp, test_extern))),
            // The generated test case
            Annotatable::Stmt(P(cx.stmt_item(sp, test_const))),
            // The original item
            Annotatable::Stmt(P(cx.stmt_item(sp, item))),
        ]
    } else {
        vec![
            // Access to libtest under a hygienic name
            Annotatable::Item(test_extern),
            // The generated test case
            Annotatable::Item(test_const),
            // The original item
            Annotatable::Item(item),
        ]
    }
}

fn item_path(mod_path: &[Ident], item_ident: &Ident) -> String {
    mod_path
        .iter()
        .chain(iter::once(item_ident))
        .map(|x| x.to_string())
        .collect::<Vec<String>>()
        .join("::")
}

enum ShouldPanic {
    No,
    Yes(Option<Symbol>),
}

fn should_ignore(sess: &Session, i: &ast::Item) -> bool {
    sess.contains_name(&i.attrs, sym::ignore)
}

fn should_fail(sess: &Session, i: &ast::Item) -> bool {
    sess.contains_name(&i.attrs, sym::allow_fail)
}

fn should_panic(cx: &ExtCtxt<'_>, i: &ast::Item) -> ShouldPanic {
    match cx.sess.find_by_name(&i.attrs, sym::should_panic) {
        Some(attr) => {
            let sd = &cx.sess.parse_sess.span_diagnostic;

            match attr.meta_item_list() {
                // Handle #[should_panic(expected = "foo")]
                Some(list) => {
                    let msg = list
                        .iter()
                        .find(|mi| mi.has_name(sym::expected))
                        .and_then(|mi| mi.meta_item())
                        .and_then(|mi| mi.value_str());
                    if list.len() != 1 || msg.is_none() {
                        sd.struct_span_warn(
                            attr.span,
                            "argument must be of the form: \
                             `expected = \"error message\"`",
                        )
                        .note(
                            "errors in this attribute were erroneously \
                                allowed and will become a hard error in a \
                                future release",
                        )
                        .emit();
                        ShouldPanic::Yes(None)
                    } else {
                        ShouldPanic::Yes(msg)
                    }
                }
                // Handle #[should_panic] and #[should_panic = "expected"]
                None => ShouldPanic::Yes(attr.value_str()),
            }
        }
        None => ShouldPanic::No,
    }
}

enum TestType {
    UnitTest,
    IntegrationTest,
    Unknown,
}

/// Attempts to determine the type of test.
/// Since doctests are created without macro expanding, only possible variants here
/// are `UnitTest`, `IntegrationTest` or `Unknown`.
fn test_type(cx: &ExtCtxt<'_>) -> TestType {
    // Root path from context contains the topmost sources directory of the crate.
    // I.e., for `project` with sources in `src` and tests in `tests` folders
    // (no matter how many nested folders lie inside),
    // there will be two different root paths: `/project/src` and `/project/tests`.
    let crate_path = cx.root_path.as_path();

    if crate_path.ends_with("src") {
        // `/src` folder contains unit-tests.
        TestType::UnitTest
    } else if crate_path.ends_with("tests") {
        // `/tests` folder contains integration tests.
        TestType::IntegrationTest
    } else {
        // Crate layout doesn't match expected one, test type is unknown.
        TestType::Unknown
    }
}

fn has_test_signature(cx: &ExtCtxt<'_>, i: &ast::Item) -> bool {
    let has_should_panic_attr = cx.sess.contains_name(&i.attrs, sym::should_panic);
    let sd = &cx.sess.parse_sess.span_diagnostic;
    if let ast::ItemKind::Fn(box ast::Fn { ref sig, ref generics, .. }) = i.kind {
        if let ast::Unsafe::Yes(span) = sig.header.unsafety {
            sd.struct_span_err(i.span, "unsafe functions cannot be used for tests")
                .span_label(span, "`unsafe` because of this")
                .emit();
            return false;
        }
        if let ast::Async::Yes { span, .. } = sig.header.asyncness {
            sd.struct_span_err(i.span, "async functions cannot be used for tests")
                .span_label(span, "`async` because of this")
                .emit();
            return false;
        }

        // If the termination trait is active, the compiler will check that the output
        // type implements the `Termination` trait as `libtest` enforces that.
        let has_output = match sig.decl.output {
            ast::FnRetTy::Default(..) => false,
            ast::FnRetTy::Ty(ref t) if t.kind.is_unit() => false,
            _ => true,
        };

        if !sig.decl.inputs.is_empty() {
            sd.span_err(i.span, "functions used as tests can not have any arguments");
            return false;
        }

        match (has_output, has_should_panic_attr) {
            (true, true) => {
                sd.span_err(i.span, "functions using `#[should_panic]` must return `()`");
                false
            }
            (true, false) => {
                if !generics.params.is_empty() {
                    sd.span_err(i.span, "functions used as tests must have signature fn() -> ()");
                    false
                } else {
                    true
                }
            }
            (false, _) => true,
        }
    } else {
        sd.span_err(i.span, "only functions may be used as tests");
        false
    }
}

fn has_bench_signature(cx: &ExtCtxt<'_>, i: &ast::Item) -> bool {
    let has_sig = if let ast::ItemKind::Fn(box ast::Fn { ref sig, .. }) = i.kind {
        // N.B., inadequate check, but we're running
        // well before resolve, can't get too deep.
        sig.decl.inputs.len() == 1
    } else {
        false
    };

    if !has_sig {
        cx.sess.parse_sess.span_diagnostic.span_err(
            i.span,
            "functions used as benches must have \
            signature `fn(&mut Bencher) -> impl Termination`",
        );
    }

    has_sig
}
