//! The expansion from a test function to the appropriate test struct for libtest
//! Ideally, this code would be in libtest but for efficiency and error messages it lives here.

use std::assert_matches::assert_matches;
use std::iter;

use rustc_ast::{self as ast, GenericParamKind, HasNodeId, attr, join_path_idents};
use rustc_ast_pretty::pprust;
use rustc_attr_parsing::AttributeParser;
use rustc_errors::{Applicability, Diag, Level};
use rustc_expand::base::*;
use rustc_hir::Attribute;
use rustc_hir::attrs::AttributeKind;
use rustc_span::{ErrorGuaranteed, FileNameDisplayPreference, Ident, Span, Symbol, sym};
use thin_vec::{ThinVec, thin_vec};
use tracing::debug;

use crate::errors;
use crate::util::{check_builtin_macro_attribute, warn_on_duplicate_attribute};

/// #[test_case] is used by custom test authors to mark tests
/// When building for test, it needs to make the item public and gensym the name
/// Otherwise, we'll omit the item. This behavior means that any item annotated
/// with #[test_case] is never addressable.
///
/// We mark item with an inert attribute "rustc_test_marker" which the test generation
/// logic will pick up on.
pub(crate) fn expand_test_case(
    ecx: &mut ExtCtxt<'_>,
    attr_sp: Span,
    meta_item: &ast::MetaItem,
    anno_item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::test_case);
    warn_on_duplicate_attribute(ecx, &anno_item, sym::test_case);

    if !ecx.ecfg.should_test {
        return vec![];
    }

    let sp = ecx.with_def_site_ctxt(attr_sp);
    let (mut item, is_stmt) = match anno_item {
        Annotatable::Item(item) => (item, false),
        Annotatable::Stmt(stmt) if let ast::StmtKind::Item(_) = stmt.kind => {
            if let ast::StmtKind::Item(i) = stmt.kind {
                (i, true)
            } else {
                unreachable!()
            }
        }
        _ => {
            ecx.dcx().emit_err(errors::TestCaseNonItem { span: anno_item.span() });
            return vec![];
        }
    };

    // `#[test_case]` is valid on functions, consts, and statics. Only modify
    // the item in those cases.
    match &mut item.kind {
        ast::ItemKind::Fn(box ast::Fn { ident, .. })
        | ast::ItemKind::Const(box ast::ConstItem { ident, .. })
        | ast::ItemKind::Static(box ast::StaticItem { ident, .. }) => {
            ident.span = ident.span.with_ctxt(sp.ctxt());
            let test_path_symbol = Symbol::intern(&item_path(
                // skip the name of the root module
                &ecx.current_expansion.module.mod_path[1..],
                ident,
            ));
            item.vis = ast::Visibility {
                span: item.vis.span,
                kind: ast::VisibilityKind::Public,
                tokens: None,
            };
            item.attrs.push(ecx.attr_name_value_str(sym::rustc_test_marker, test_path_symbol, sp));
        }
        _ => {}
    }

    let ret = if is_stmt {
        Annotatable::Stmt(Box::new(ecx.stmt_item(item.span, item)))
    } else {
        Annotatable::Item(item)
    };

    vec![ret]
}

pub(crate) fn expand_test(
    cx: &mut ExtCtxt<'_>,
    attr_sp: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(cx, meta_item, sym::test);
    warn_on_duplicate_attribute(cx, &item, sym::test);
    expand_test_or_bench(cx, attr_sp, item, false)
}

pub(crate) fn expand_bench(
    cx: &mut ExtCtxt<'_>,
    attr_sp: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(cx, meta_item, sym::bench);
    warn_on_duplicate_attribute(cx, &item, sym::bench);
    expand_test_or_bench(cx, attr_sp, item, true)
}

pub(crate) fn expand_test_or_bench(
    cx: &ExtCtxt<'_>,
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
        Annotatable::Stmt(box ast::Stmt { kind: ast::StmtKind::Item(i), .. }) => (i, true),
        other => {
            not_testable_error(cx, attr_sp, None);
            return vec![other];
        }
    };

    let ast::ItemKind::Fn(fn_) = &item.kind else {
        not_testable_error(cx, attr_sp, Some(&item));
        return if is_stmt {
            vec![Annotatable::Stmt(Box::new(cx.stmt_item(item.span, item)))]
        } else {
            vec![Annotatable::Item(item)]
        };
    };

    if let Some(attr) = attr::find_by_name(&item.attrs, sym::naked) {
        cx.dcx().emit_err(errors::NakedFunctionTestingAttribute {
            testing_span: attr_sp,
            naked_span: attr.span,
        });
        return vec![Annotatable::Item(item)];
    }

    // check_*_signature will report any errors in the type so compilation
    // will fail. We shouldn't try to expand in this case because the errors
    // would be spurious.
    let check_result = if is_bench {
        check_bench_signature(cx, &item, fn_)
    } else {
        check_test_signature(cx, &item, fn_)
    };
    if check_result.is_err() {
        return if is_stmt {
            vec![Annotatable::Stmt(Box::new(cx.stmt_item(item.span, item)))]
        } else {
            vec![Annotatable::Item(item)]
        };
    }

    let sp = cx.with_def_site_ctxt(item.span);
    let ret_ty_sp = cx.with_def_site_ctxt(fn_.sig.decl.output.span());
    let attr_sp = cx.with_def_site_ctxt(attr_sp);

    let test_ident = Ident::new(sym::test, attr_sp);

    // creates test::$name
    let test_path = |name| cx.path(ret_ty_sp, vec![test_ident, Ident::from_str_and_span(name, sp)]);

    // creates test::ShouldPanic::$name
    let should_panic_path = |name| {
        cx.path(
            sp,
            vec![
                test_ident,
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
                test_ident,
                Ident::from_str_and_span("TestType", sp),
                Ident::from_str_and_span(name, sp),
            ],
        )
    };

    // creates $name: $expr
    let field = |name, expr| cx.field_imm(sp, Ident::from_str_and_span(name, sp), expr);

    // Adds `#[coverage(off)]` to a closure, so it won't be instrumented in
    // `-Cinstrument-coverage` builds.
    // This requires `#[allow_internal_unstable(coverage_attribute)]` on the
    // corresponding macro declaration in `core::macros`.
    let coverage_off = |mut expr: Box<ast::Expr>| {
        assert_matches!(expr.kind, ast::ExprKind::Closure(_));
        expr.attrs.push(cx.attr_nested_word(sym::coverage, sym::off, sp));
        expr
    };

    let test_fn = if is_bench {
        // A simple ident for a lambda
        let b = Ident::from_str_and_span("b", attr_sp);

        cx.expr_call(
            sp,
            cx.expr_path(test_path("StaticBenchFn")),
            thin_vec![
                // #[coverage(off)]
                // |b| self::test::assert_test_result(
                coverage_off(cx.lambda1(
                    sp,
                    cx.expr_call(
                        sp,
                        cx.expr_path(test_path("assert_test_result")),
                        thin_vec![
                            // super::$test_fn(b)
                            cx.expr_call(
                                ret_ty_sp,
                                cx.expr_path(cx.path(sp, vec![fn_.ident])),
                                thin_vec![cx.expr_ident(sp, b)],
                            ),
                        ],
                    ),
                    b,
                )), // )
            ],
        )
    } else {
        cx.expr_call(
            sp,
            cx.expr_path(test_path("StaticTestFn")),
            thin_vec![
                // #[coverage(off)]
                // || {
                coverage_off(cx.lambda0(
                    sp,
                    // test::assert_test_result(
                    cx.expr_call(
                        sp,
                        cx.expr_path(test_path("assert_test_result")),
                        thin_vec![
                            // $test_fn()
                            cx.expr_call(
                                ret_ty_sp,
                                cx.expr_path(cx.path(sp, vec![fn_.ident])),
                                ThinVec::new(),
                            ), // )
                        ],
                    ), // }
                )), // )
            ],
        )
    };

    let test_path_symbol = Symbol::intern(&item_path(
        // skip the name of the root module
        &cx.current_expansion.module.mod_path[1..],
        &fn_.ident,
    ));

    let location_info = get_location_info(cx, &fn_);

    let mut test_const =
        cx.item(
            sp,
            thin_vec![
                // #[cfg(test)]
                cx.attr_nested_word(sym::cfg, sym::test, attr_sp),
                // #[rustc_test_marker = "test_case_sort_key"]
                cx.attr_name_value_str(sym::rustc_test_marker, test_path_symbol, attr_sp),
                // #[doc(hidden)]
                cx.attr_nested_word(sym::doc, sym::hidden, attr_sp),
            ],
            // const $ident: test::TestDescAndFn =
            ast::ItemKind::Const(
                ast::ConstItem {
                    defaultness: ast::Defaultness::Final,
                    ident: Ident::new(fn_.ident.name, sp),
                    generics: ast::Generics::default(),
                    ty: cx.ty(sp, ast::TyKind::Path(None, test_path("TestDescAndFn"))),
                    define_opaque: None,
                    // test::TestDescAndFn {
                    expr: Some(
                        cx.expr_struct(
                            sp,
                            test_path("TestDescAndFn"),
                            thin_vec![
                        // desc: test::TestDesc {
                        field(
                            "desc",
                            cx.expr_struct(sp, test_path("TestDesc"), thin_vec![
                                // name: "path::to::test"
                                field(
                                    "name",
                                    cx.expr_call(
                                        sp,
                                        cx.expr_path(test_path("StaticTestName")),
                                        thin_vec![cx.expr_str(sp, test_path_symbol)],
                                    ),
                                ),
                                // ignore: true | false
                                field("ignore", cx.expr_bool(sp, should_ignore(&item)),),
                                // ignore_message: Some("...") | None
                                field(
                                    "ignore_message",
                                    if let Some(msg) = should_ignore_message(&item) {
                                        cx.expr_some(sp, cx.expr_str(sp, msg))
                                    } else {
                                        cx.expr_none(sp)
                                    },
                                ),
                                // source_file: <relative_path_of_source_file>
                                field("source_file", cx.expr_str(sp, location_info.0)),
                                // start_line: start line of the test fn identifier.
                                field("start_line", cx.expr_usize(sp, location_info.1)),
                                // start_col: start column of the test fn identifier.
                                field("start_col", cx.expr_usize(sp, location_info.2)),
                                // end_line: end line of the test fn identifier.
                                field("end_line", cx.expr_usize(sp, location_info.3)),
                                // end_col: end column of the test fn identifier.
                                field("end_col", cx.expr_usize(sp, location_info.4)),
                                // compile_fail: true | false
                                field("compile_fail", cx.expr_bool(sp, false)),
                                // no_run: true | false
                                field("no_run", cx.expr_bool(sp, false)),
                                // should_panic: ...
                                field("should_panic", match should_panic(cx, &item) {
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
                                        thin_vec![cx.expr_str(sp, sym)],
                                    ),
                                },),
                                // test_type: ...
                                field("test_type", match test_type(cx) {
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
                                },),
                                // },
                            ],),
                        ),
                        // testfn: test::StaticTestFn(...) | test::StaticBenchFn(...)
                        field("testfn", test_fn), // }
                    ],
                        ), // }
                    ),
                }
                .into(),
            ),
        );
    test_const.vis.kind = ast::VisibilityKind::Public;

    // extern crate test
    let test_extern =
        cx.item(sp, ast::AttrVec::new(), ast::ItemKind::ExternCrate(None, test_ident));

    debug!("synthetic test item:\n{}\n", pprust::item_to_string(&test_const));

    if is_stmt {
        vec![
            // Access to libtest under a hygienic name
            Annotatable::Stmt(Box::new(cx.stmt_item(sp, test_extern))),
            // The generated test case
            Annotatable::Stmt(Box::new(cx.stmt_item(sp, test_const))),
            // The original item
            Annotatable::Stmt(Box::new(cx.stmt_item(sp, item))),
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

fn not_testable_error(cx: &ExtCtxt<'_>, attr_sp: Span, item: Option<&ast::Item>) {
    let dcx = cx.dcx();
    let msg = "the `#[test]` attribute may only be used on a non-associated function";
    let level = match item.map(|i| &i.kind) {
        // These were a warning before #92959 and need to continue being that to avoid breaking
        // stable user code (#94508).
        Some(ast::ItemKind::MacCall(_)) => Level::Warning,
        _ => Level::Error,
    };
    let mut err = Diag::<()>::new(dcx, level, msg);
    err.span(attr_sp);
    if let Some(item) = item {
        err.span_label(
            item.span,
            format!(
                "expected a non-associated function, found {} {}",
                item.kind.article(),
                item.kind.descr()
            ),
        );
    }
    err.with_span_label(attr_sp, "the `#[test]` macro causes a function to be run as a test and has no effect on non-functions")
        .with_span_suggestion(attr_sp,
            "replace with conditional compilation to make the item only exist when tests are being run",
            "#[cfg(test)]",
            Applicability::MaybeIncorrect)
        .emit();
}

fn get_location_info(cx: &ExtCtxt<'_>, fn_: &ast::Fn) -> (Symbol, usize, usize, usize, usize) {
    let span = fn_.ident.span;
    let (source_file, lo_line, lo_col, hi_line, hi_col) =
        cx.sess.source_map().span_to_location_info(span);

    let file_name = match source_file {
        Some(sf) => sf.name.display(FileNameDisplayPreference::Remapped).to_string(),
        None => "no-location".to_string(),
    };

    (Symbol::intern(&file_name), lo_line, lo_col, hi_line, hi_col)
}

fn item_path(mod_path: &[Ident], item_ident: &Ident) -> String {
    join_path_idents(mod_path.iter().chain(iter::once(item_ident)))
}

enum ShouldPanic {
    No,
    Yes(Option<Symbol>),
}

fn should_ignore(i: &ast::Item) -> bool {
    attr::contains_name(&i.attrs, sym::ignore)
}

fn should_ignore_message(i: &ast::Item) -> Option<Symbol> {
    match attr::find_by_name(&i.attrs, sym::ignore) {
        Some(attr) => {
            match attr.meta_item_list() {
                // Handle #[ignore(bar = "foo")]
                Some(_) => None,
                // Handle #[ignore] and #[ignore = "message"]
                None => attr.value_str(),
            }
        }
        None => None,
    }
}

fn should_panic(cx: &ExtCtxt<'_>, i: &ast::Item) -> ShouldPanic {
    if let Some(Attribute::Parsed(AttributeKind::ShouldPanic { reason, .. })) =
        AttributeParser::parse_limited(
            cx.sess,
            &i.attrs,
            sym::should_panic,
            i.span,
            i.node_id(),
            None,
        )
    {
        ShouldPanic::Yes(reason)
    } else {
        ShouldPanic::No
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

fn check_test_signature(
    cx: &ExtCtxt<'_>,
    i: &ast::Item,
    f: &ast::Fn,
) -> Result<(), ErrorGuaranteed> {
    let has_should_panic_attr = attr::contains_name(&i.attrs, sym::should_panic);
    let dcx = cx.dcx();

    if let ast::Safety::Unsafe(span) = f.sig.header.safety {
        return Err(dcx.emit_err(errors::TestBadFn { span: i.span, cause: span, kind: "unsafe" }));
    }

    if let Some(coroutine_kind) = f.sig.header.coroutine_kind {
        match coroutine_kind {
            ast::CoroutineKind::Async { span, .. } => {
                return Err(dcx.emit_err(errors::TestBadFn {
                    span: i.span,
                    cause: span,
                    kind: "async",
                }));
            }
            ast::CoroutineKind::Gen { span, .. } => {
                return Err(dcx.emit_err(errors::TestBadFn {
                    span: i.span,
                    cause: span,
                    kind: "gen",
                }));
            }
            ast::CoroutineKind::AsyncGen { span, .. } => {
                return Err(dcx.emit_err(errors::TestBadFn {
                    span: i.span,
                    cause: span,
                    kind: "async gen",
                }));
            }
        }
    }

    // If the termination trait is active, the compiler will check that the output
    // type implements the `Termination` trait as `libtest` enforces that.
    let has_output = match &f.sig.decl.output {
        ast::FnRetTy::Default(..) => false,
        ast::FnRetTy::Ty(t) if t.kind.is_unit() => false,
        _ => true,
    };

    if !f.sig.decl.inputs.is_empty() {
        return Err(dcx.span_err(i.span, "functions used as tests can not have any arguments"));
    }

    if has_should_panic_attr && has_output {
        return Err(dcx.span_err(i.span, "functions using `#[should_panic]` must return `()`"));
    }

    if f.generics.params.iter().any(|param| !matches!(param.kind, GenericParamKind::Lifetime)) {
        return Err(dcx.span_err(
            i.span,
            "functions used as tests can not have any non-lifetime generic parameters",
        ));
    }

    Ok(())
}

fn check_bench_signature(
    cx: &ExtCtxt<'_>,
    i: &ast::Item,
    f: &ast::Fn,
) -> Result<(), ErrorGuaranteed> {
    // N.B., inadequate check, but we're running
    // well before resolve, can't get too deep.
    if f.sig.decl.inputs.len() != 1 {
        return Err(cx.dcx().emit_err(errors::BenchSig { span: i.span }));
    }
    Ok(())
}
