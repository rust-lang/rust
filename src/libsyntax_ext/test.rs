/// The expansion from a test function to the appropriate test struct for libtest
/// Ideally, this code would be in libtest but for efficiency and error messages it lives here.

use syntax::ext::base::*;
use syntax::ext::build::AstBuilder;
use syntax::ext::hygiene::{Mark, SyntaxContext};
use syntax::attr;
use syntax::ast;
use syntax::print::pprust;
use syntax::symbol::{Symbol, sym};
use syntax_pos::Span;
use syntax::source_map::{ExpnInfo, MacroAttribute};
use std::iter;

pub fn expand_test(
    cx: &mut ExtCtxt<'_>,
    attr_sp: Span,
    _meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    expand_test_or_bench(cx, attr_sp, item, false)
}

pub fn expand_bench(
    cx: &mut ExtCtxt<'_>,
    attr_sp: Span,
    _meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    expand_test_or_bench(cx, attr_sp, item, true)
}

pub fn expand_test_or_bench(
    cx: &mut ExtCtxt<'_>,
    attr_sp: Span,
    item: Annotatable,
    is_bench: bool
) -> Vec<Annotatable> {
    // If we're not in test configuration, remove the annotated item
    if !cx.ecfg.should_test { return vec![]; }

    let item =
        if let Annotatable::Item(i) = item { i }
        else {
            cx.parse_sess.span_diagnostic.span_fatal(item.span(),
                "#[test] attribute is only allowed on non associated functions").raise();
        };

    if let ast::ItemKind::Mac(_) = item.node {
        cx.parse_sess.span_diagnostic.span_warn(item.span,
            "#[test] attribute should not be used on macros. Use #[cfg(test)] instead.");
        return vec![Annotatable::Item(item)];
    }

    // has_*_signature will report any errors in the type so compilation
    // will fail. We shouldn't try to expand in this case because the errors
    // would be spurious.
    if (!is_bench && !has_test_signature(cx, &item)) ||
        (is_bench && !has_bench_signature(cx, &item)) {
        return vec![Annotatable::Item(item)];
    }

    let (sp, attr_sp) = {
        let mark = Mark::fresh(Mark::root());
        mark.set_expn_info(ExpnInfo::with_unstable(
            MacroAttribute(sym::test), attr_sp, cx.parse_sess.edition,
            &[sym::rustc_attrs, sym::test],
        ));
        (item.span.with_ctxt(SyntaxContext::empty().apply_mark(mark)),
         attr_sp.with_ctxt(SyntaxContext::empty().apply_mark(mark)))
    };

    // Gensym "test" so we can extern crate without conflicting with any local names
    let test_id = cx.ident_of("test").gensym();

    // creates test::$name
    let test_path = |name| {
        cx.path(sp, vec![test_id, cx.ident_of(name)])
    };

    // creates test::ShouldPanic::$name
    let should_panic_path = |name| {
        cx.path(sp, vec![test_id, cx.ident_of("ShouldPanic"), cx.ident_of(name)])
    };

    // creates $name: $expr
    let field = |name, expr| cx.field_imm(sp, cx.ident_of(name), expr);

    let test_fn = if is_bench {
        // A simple ident for a lambda
        let b = cx.ident_of("b");

        cx.expr_call(sp, cx.expr_path(test_path("StaticBenchFn")), vec![
            // |b| self::test::assert_test_result(
            cx.lambda1(sp,
                cx.expr_call(sp, cx.expr_path(test_path("assert_test_result")), vec![
                    // super::$test_fn(b)
                    cx.expr_call(sp,
                        cx.expr_path(cx.path(sp, vec![item.ident])),
                        vec![cx.expr_ident(sp, b)])
                ]),
                b
            )
            // )
        ])
    } else {
        cx.expr_call(sp, cx.expr_path(test_path("StaticTestFn")), vec![
            // || {
            cx.lambda0(sp,
                // test::assert_test_result(
                cx.expr_call(sp, cx.expr_path(test_path("assert_test_result")), vec![
                    // $test_fn()
                    cx.expr_call(sp, cx.expr_path(cx.path(sp, vec![item.ident])), vec![])
                // )
                ])
            // }
            )
        // )
        ])
    };

    let mut test_const = cx.item(sp, ast::Ident::new(item.ident.name, sp).gensym(),
        vec![
            // #[cfg(test)]
            cx.attribute(attr_sp, cx.meta_list(attr_sp, sym::cfg, vec![
                cx.meta_list_item_word(attr_sp, sym::test)
            ])),
            // #[rustc_test_marker]
            cx.attribute(attr_sp, cx.meta_word(attr_sp, sym::rustc_test_marker)),
        ],
        // const $ident: test::TestDescAndFn =
        ast::ItemKind::Const(cx.ty(sp, ast::TyKind::Path(None, test_path("TestDescAndFn"))),
            // test::TestDescAndFn {
            cx.expr_struct(sp, test_path("TestDescAndFn"), vec![
                // desc: test::TestDesc {
                field("desc", cx.expr_struct(sp, test_path("TestDesc"), vec![
                    // name: "path::to::test"
                    field("name", cx.expr_call(sp, cx.expr_path(test_path("StaticTestName")),
                        vec![
                            cx.expr_str(sp, Symbol::intern(&item_path(
                                // skip the name of the root module
                                &cx.current_expansion.module.mod_path[1..],
                                &item.ident
                            )))
                        ])),
                    // ignore: true | false
                    field("ignore", cx.expr_bool(sp, should_ignore(&item))),
                    // allow_fail: true | false
                    field("allow_fail", cx.expr_bool(sp, should_fail(&item))),
                    // should_panic: ...
                    field("should_panic", match should_panic(cx, &item) {
                        // test::ShouldPanic::No
                        ShouldPanic::No => cx.expr_path(should_panic_path("No")),
                        // test::ShouldPanic::Yes
                        ShouldPanic::Yes(None) => cx.expr_path(should_panic_path("Yes")),
                        // test::ShouldPanic::YesWithMessage("...")
                        ShouldPanic::Yes(Some(sym)) => cx.expr_call(sp,
                            cx.expr_path(should_panic_path("YesWithMessage")),
                            vec![cx.expr_str(sp, sym)]),
                    }),
                // },
                ])),
                // testfn: test::StaticTestFn(...) | test::StaticBenchFn(...)
                field("testfn", test_fn)
            // }
            ])
        // }
        ));
    test_const = test_const.map(|mut tc| { tc.vis.node = ast::VisibilityKind::Public; tc});

    // extern crate test as test_gensym
    let test_extern = cx.item(sp,
        test_id,
        vec![],
        ast::ItemKind::ExternCrate(Some(sym::test))
    );

    log::debug!("Synthetic test item:\n{}\n", pprust::item_to_string(&test_const));

    vec![
        // Access to libtest under a gensymed name
        Annotatable::Item(test_extern),
        // The generated test case
        Annotatable::Item(test_const),
        // The original item
        Annotatable::Item(item)
    ]
}

fn item_path(mod_path: &[ast::Ident], item_ident: &ast::Ident) -> String {
    mod_path.iter().chain(iter::once(item_ident))
        .map(|x| x.to_string()).collect::<Vec<String>>().join("::")
}

enum ShouldPanic {
    No,
    Yes(Option<Symbol>),
}

fn should_ignore(i: &ast::Item) -> bool {
    attr::contains_name(&i.attrs, sym::ignore)
}

fn should_fail(i: &ast::Item) -> bool {
    attr::contains_name(&i.attrs, sym::allow_fail)
}

fn should_panic(cx: &ExtCtxt<'_>, i: &ast::Item) -> ShouldPanic {
    match attr::find_by_name(&i.attrs, sym::should_panic) {
        Some(attr) => {
            let ref sd = cx.parse_sess.span_diagnostic;

            match attr.meta_item_list() {
                // Handle #[should_panic(expected = "foo")]
                Some(list) => {
                    let msg = list.iter()
                        .find(|mi| mi.check_name(sym::expected))
                        .and_then(|mi| mi.meta_item())
                        .and_then(|mi| mi.value_str());
                    if list.len() != 1 || msg.is_none() {
                        sd.struct_span_warn(
                            attr.span,
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
                // Handle #[should_panic] and #[should_panic = "expected"]
                None => ShouldPanic::Yes(attr.value_str())
            }
        }
        None => ShouldPanic::No,
    }
}

fn has_test_signature(cx: &ExtCtxt<'_>, i: &ast::Item) -> bool {
    let has_should_panic_attr = attr::contains_name(&i.attrs, sym::should_panic);
    let ref sd = cx.parse_sess.span_diagnostic;
    if let ast::ItemKind::Fn(ref decl, ref header, ref generics, _) = i.node {
        if header.unsafety == ast::Unsafety::Unsafe {
            sd.span_err(
                i.span,
                "unsafe functions cannot be used for tests"
            );
            return false
        }
        if header.asyncness.node.is_async() {
            sd.span_err(
                i.span,
                "async functions cannot be used for tests"
            );
            return false
        }


        // If the termination trait is active, the compiler will check that the output
        // type implements the `Termination` trait as `libtest` enforces that.
        let has_output = match decl.output {
            ast::FunctionRetTy::Default(..) => false,
            ast::FunctionRetTy::Ty(ref t) if t.node.is_unit() => false,
            _ => true
        };

        if !decl.inputs.is_empty() {
            sd.span_err(i.span, "functions used as tests can not have any arguments");
            return false;
        }

        match (has_output, has_should_panic_attr) {
            (true, true) => {
                sd.span_err(i.span, "functions using `#[should_panic]` must return `()`");
                false
            },
            (true, false) => if !generics.params.is_empty() {
                sd.span_err(i.span,
                                "functions used as tests must have signature fn() -> ()");
                false
            } else {
                true
            },
            (false, _) => true
        }
    } else {
        sd.span_err(i.span, "only functions may be used as tests");
        false
    }
}

fn has_bench_signature(cx: &ExtCtxt<'_>, i: &ast::Item) -> bool {
    let has_sig = if let ast::ItemKind::Fn(ref decl, _, _, _) = i.node {
        // N.B., inadequate check, but we're running
        // well before resolve, can't get too deep.
        decl.inputs.len() == 1
    } else {
        false
    };

    if !has_sig {
        cx.parse_sess.span_diagnostic.span_err(i.span, "functions used as benches must have \
            signature `fn(&mut Bencher) -> impl Termination`");
    }

    has_sig
}
