//! Implementation of "type" inlay hints:
//! ```no_run
//! fn f(a: i32, b: i32) -> i32 { a + b }
//! let _x /* i32 */= f(4, 4);
//! ```
use hir::{DisplayTarget, Semantics};
use ide_db::{RootDatabase, famous_defs::FamousDefs};

use itertools::Itertools;
use syntax::{
    ast::{self, AstNode, HasGenericArgs, HasName},
    match_ast,
};

use crate::{
    InlayHint, InlayHintPosition, InlayHintsConfig, InlayKind,
    inlay_hints::{closure_has_block_body, label_of_ty, ty_to_text_edit},
};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    famous_defs @ FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    display_target: DisplayTarget,
    pat: &ast::IdentPat,
) -> Option<()> {
    if !config.type_hints {
        return None;
    }

    let parent = pat.syntax().parent()?;
    let type_ascriptable = match_ast! {
        match parent {
            ast::Param(it) => {
                if it.ty().is_some() {
                    return None;
                }
                if config.hide_closure_parameter_hints && it.syntax().ancestors().nth(2).is_none_or(|n| matches!(ast::Expr::cast(n), Some(ast::Expr::ClosureExpr(_)))) {
                    return None;
                }
                Some(it.colon_token())
            },
            ast::LetStmt(it) => {
                if config.hide_closure_initialization_hints {
                    if let Some(ast::Expr::ClosureExpr(closure)) = it.initializer() {
                        if closure_has_block_body(&closure) {
                            return None;
                        }
                    }
                }
                if it.ty().is_some() {
                    return None;
                }
                Some(it.colon_token())
            },
            _ => None
        }
    };

    let descended = sema.descend_node_into_attributes(pat.clone()).pop();
    let desc_pat = descended.as_ref().unwrap_or(pat);
    let ty = sema.type_of_binding_in_pat(desc_pat)?;

    if ty.is_unknown() {
        return None;
    }

    if sema.resolve_bind_pat_to_const(pat).is_some() {
        return None;
    }

    let mut label = label_of_ty(famous_defs, config, &ty, display_target)?;

    if config.hide_named_constructor_hints
        && is_named_constructor(sema, pat, &label.to_string()).is_some()
    {
        return None;
    }

    let text_edit = if let Some(colon_token) = &type_ascriptable {
        ty_to_text_edit(
            sema,
            config,
            desc_pat.syntax(),
            &ty,
            colon_token
                .as_ref()
                .map_or_else(|| pat.syntax().text_range(), |t| t.text_range())
                .end(),
            &|_| (),
            if colon_token.is_some() { "" } else { ": " },
        )
    } else {
        None
    };

    let render_colons = config.render_colons && !matches!(type_ascriptable, Some(Some(_)));
    if render_colons {
        label.prepend_str(": ");
    }

    let text_range = match pat.name() {
        Some(name) => name.syntax().text_range(),
        None => pat.syntax().text_range(),
    };
    acc.push(InlayHint {
        range: match type_ascriptable {
            Some(Some(t)) => text_range.cover(t.text_range()),
            _ => text_range,
        },
        kind: InlayKind::Type,
        label,
        text_edit,
        position: InlayHintPosition::After,
        pad_left: !render_colons,
        pad_right: false,
        resolve_parent: Some(pat.syntax().text_range()),
    });

    Some(())
}

fn is_named_constructor(
    sema: &Semantics<'_, RootDatabase>,
    pat: &ast::IdentPat,
    ty_name: &str,
) -> Option<()> {
    let let_node = pat.syntax().parent()?;
    let expr = match_ast! {
        match let_node {
            ast::LetStmt(it) => it.initializer(),
            ast::LetExpr(it) => it.expr(),
            _ => None,
        }
    }?;

    let expr = sema.descend_node_into_attributes(expr.clone()).pop().unwrap_or(expr);
    // unwrap postfix expressions
    let expr = match expr {
        ast::Expr::TryExpr(it) => it.expr(),
        ast::Expr::AwaitExpr(it) => it.expr(),
        expr => Some(expr),
    }?;
    let expr = match expr {
        ast::Expr::CallExpr(call) => match call.expr()? {
            ast::Expr::PathExpr(path) => path,
            _ => return None,
        },
        ast::Expr::PathExpr(path) => path,
        _ => return None,
    };
    let path = expr.path()?;

    let callable = sema.type_of_expr(&ast::Expr::PathExpr(expr))?.original.as_callable(sema.db);
    let callable_kind = callable.map(|it| it.kind());
    let qual_seg = match callable_kind {
        Some(hir::CallableKind::Function(_) | hir::CallableKind::TupleEnumVariant(_)) => {
            path.qualifier()?.segment()
        }
        _ => path.segment(),
    }?;

    let ctor_name = match qual_seg.kind()? {
        ast::PathSegmentKind::Name(name_ref) => {
            match qual_seg.generic_arg_list().map(|it| it.generic_args()) {
                Some(generics) => format!("{name_ref}<{}>", generics.format(", ")),
                None => name_ref.to_string(),
            }
        }
        ast::PathSegmentKind::Type { type_ref: Some(ty), trait_ref: None } => ty.to_string(),
        _ => return None,
    };
    (ctor_name == ty_name).then_some(())
}

#[cfg(test)]
mod tests {
    // This module also contains tests for super::closure_ret

    use expect_test::expect;
    use hir::ClosureStyle;
    use syntax::{TextRange, TextSize};
    use test_utils::extract_annotations;

    use crate::{ClosureReturnTypeHints, fixture, inlay_hints::InlayHintsConfig};

    use crate::inlay_hints::tests::{
        DISABLED_CONFIG, TEST_CONFIG, check, check_edit, check_no_edit, check_with_config,
    };

    #[track_caller]
    fn check_types(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        check_with_config(InlayHintsConfig { type_hints: true, ..DISABLED_CONFIG }, ra_fixture);
    }

    #[test]
    fn type_hints_only() {
        check_types(
            r#"
fn foo(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _x = foo(4, 4);
      //^^ i32
}"#,
        );
    }

    #[test]
    fn type_hints_bindings_after_at() {
        check_types(
            r#"
//- minicore: option
fn main() {
    let ref foo @ bar @ ref mut baz = 0;
          //^^^ &i32
                //^^^ i32
                              //^^^ &mut i32
    let [x @ ..] = [0];
       //^ [i32; 1]
    if let x @ Some(_) = Some(0) {}
         //^ Option<i32>
    let foo @ (bar, baz) = (3, 3);
      //^^^ (i32, i32)
             //^^^ i32
                  //^^^ i32
}"#,
        );
    }

    #[test]
    fn default_generic_types_should_not_be_displayed() {
        check(
            r#"
struct Test<K, T = u8> { k: K, t: T }

fn main() {
    let zz = Test { t: 23u8, k: 33 };
      //^^ Test<i32>
    let zz_ref = &zz;
      //^^^^^^ &Test<i32>
    let test = || zz;
      //^^^^ impl FnOnce() -> Test<i32>
}"#,
        );
    }

    #[test]
    fn shorten_iterators_in_associated_params() {
        check_types(
            r#"
//- minicore: iterators
use core::iter;

pub struct SomeIter<T> {}

impl<T> SomeIter<T> {
    pub fn new() -> Self { SomeIter {} }
    pub fn push(&mut self, t: T) {}
}

impl<T> Iterator for SomeIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() {
    let mut some_iter = SomeIter::new();
          //^^^^^^^^^ SomeIter<Take<Repeat<i32>>>
      some_iter.push(iter::repeat(2).take(2));
    let iter_of_iters = some_iter.take(2);
      //^^^^^^^^^^^^^ impl Iterator<Item = impl Iterator<Item = i32>>
}
"#,
        );
    }

    #[test]
    fn iterator_hint_regression_issue_12674() {
        // Ensure we don't crash while solving the projection type of iterators.
        let (analysis, file_id) = fixture::file(
            r#"
//- minicore: iterators
struct S<T>(T);
impl<T> S<T> {
    fn iter(&self) -> Iter<'_, T> { loop {} }
}
struct Iter<'a, T: 'a>(&'a T);
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> { loop {} }
}
struct Container<'a> {
    elements: S<&'a str>,
}
struct SliceIter<'a, T>(&'a T);
impl<'a, T> Iterator for SliceIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> { loop {} }
}

fn main(a: SliceIter<'_, Container>) {
    a
        .filter_map(|c| Some(c.elements.iter().filter_map(|v| Some(v))))
        .map(|e| e);
}
"#,
        );
        analysis
            .inlay_hints(
                &InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
                file_id,
                None,
            )
            .unwrap();
    }

    #[test]
    fn infer_call_method_return_associated_types_with_generic() {
        check_types(
            r#"
            pub trait Default {
                fn default() -> Self;
            }
            pub trait Foo {
                type Bar: Default;
            }

            pub fn quux<T: Foo>() -> T::Bar {
                let y = Default::default();
                  //^ <T as Foo>::Bar

                y
            }
            "#,
        );
    }

    #[test]
    fn lt_hints() {
        check_types(
            r#"
struct S<'lt>;

fn f<'a>() {
    let x = S::<'static>;
      //^ S<'static>
    let y = S::<'_>;
      //^ S<'_>
    let z = S::<'a>;
      //^ S<'a>

}
"#,
        );
    }

    #[test]
    fn fn_hints() {
        check_types(
            r#"
//- minicore: fn, sized
fn foo() -> impl Fn() { loop {} }
fn foo1() -> impl Fn(f64) { loop {} }
fn foo2() -> impl Fn(f64, f64) { loop {} }
fn foo3() -> impl Fn(f64, f64) -> u32 { loop {} }
fn foo4() -> &'static dyn Fn(f64, f64) -> u32 { loop {} }
fn foo5() -> &'static for<'a> dyn Fn(&'a dyn Fn(f64, f64) -> u32, f64) -> u32 { loop {} }
fn foo6() -> impl Fn(f64, f64) -> u32 + Sized { loop {} }
fn foo7() -> *const (impl Fn(f64, f64) -> u32 + Sized) { loop {} }

fn main() {
    let foo = foo();
     // ^^^ impl Fn()
    let foo = foo1();
     // ^^^ impl Fn(f64)
    let foo = foo2();
     // ^^^ impl Fn(f64, f64)
    let foo = foo3();
     // ^^^ impl Fn(f64, f64) -> u32
    let foo = foo4();
     // ^^^ &'static dyn Fn(f64, f64) -> u32
    let foo = foo5();
     // ^^^ &'static dyn Fn(&dyn Fn(f64, f64) -> u32, f64) -> u32
    let foo = foo6();
     // ^^^ impl Fn(f64, f64) -> u32
    let foo = foo7();
     // ^^^ *const impl Fn(f64, f64) -> u32
}
"#,
        )
    }

    #[test]
    fn check_hint_range_limit() {
        let fixture = r#"
//- minicore: fn, sized
fn foo() -> impl Fn() { loop {} }
fn foo1() -> impl Fn(f64) { loop {} }
fn foo2() -> impl Fn(f64, f64) { loop {} }
fn foo3() -> impl Fn(f64, f64) -> u32 { loop {} }
fn foo4() -> &'static dyn Fn(f64, f64) -> u32 { loop {} }
fn foo5() -> &'static dyn Fn(&'static dyn Fn(f64, f64) -> u32, f64) -> u32 { loop {} }
fn foo6() -> impl Fn(f64, f64) -> u32 + Sized { loop {} }
fn foo7() -> *const (impl Fn(f64, f64) -> u32 + Sized) { loop {} }

fn main() {
    let foo = foo();
    let foo = foo1();
    let foo = foo2();
     // ^^^ impl Fn(f64, f64)
    let foo = foo3();
     // ^^^ impl Fn(f64, f64) -> u32
    let foo = foo4();
     // ^^^ &'static dyn Fn(f64, f64) -> u32
    let foo = foo5();
    let foo = foo6();
    let foo = foo7();
}
"#;
        let (analysis, file_id) = fixture::file(fixture);
        let expected = extract_annotations(&analysis.file_text(file_id).unwrap());
        let inlay_hints = analysis
            .inlay_hints(
                &InlayHintsConfig { type_hints: true, ..DISABLED_CONFIG },
                file_id,
                Some(TextRange::new(TextSize::from(491), TextSize::from(640))),
            )
            .unwrap();
        let actual =
            inlay_hints.into_iter().map(|it| (it.range, it.label.to_string())).collect::<Vec<_>>();
        assert_eq!(expected, actual, "\nExpected:\n{expected:#?}\n\nActual:\n{actual:#?}");
    }

    #[test]
    fn fn_hints_ptr_rpit_fn_parentheses() {
        check_types(
            r#"
//- minicore: fn, sized
trait Trait {}

fn foo1() -> *const impl Fn() { loop {} }
fn foo2() -> *const (impl Fn() + Sized) { loop {} }
fn foo3() -> *const (impl Fn() + ?Sized) { loop {} }
fn foo4() -> *const (impl Sized + Fn()) { loop {} }
fn foo5() -> *const (impl ?Sized + Fn()) { loop {} }
fn foo6() -> *const (impl Fn() + Trait) { loop {} }
fn foo7() -> *const (impl Fn() + Sized + Trait) { loop {} }
fn foo8() -> *const (impl Fn() + ?Sized + Trait) { loop {} }
fn foo9() -> *const (impl Fn() -> u8 + ?Sized) { loop {} }
fn foo10() -> *const (impl Fn() + Sized + ?Sized) { loop {} }

fn main() {
    let foo = foo1();
    //  ^^^ *const impl Fn()
    let foo = foo2();
    //  ^^^ *const impl Fn()
    let foo = foo3();
    //  ^^^ *const (impl Fn() + ?Sized)
    let foo = foo4();
    //  ^^^ *const impl Fn()
    let foo = foo5();
    //  ^^^ *const (impl Fn() + ?Sized)
    let foo = foo6();
    //  ^^^ *const (impl Fn() + Trait)
    let foo = foo7();
    //  ^^^ *const (impl Fn() + Trait)
    let foo = foo8();
    //  ^^^ *const (impl Fn() + Trait + ?Sized)
    let foo = foo9();
    //  ^^^ *const (impl Fn() -> u8 + ?Sized)
    let foo = foo10();
    //  ^^^ *const impl Fn()
}
"#,
        )
    }

    #[test]
    fn unit_structs_have_no_type_hints() {
        check_types(
            r#"
//- minicore: result
struct SyntheticSyntax;

fn main() {
    match Ok(()) {
        Ok(_) => (),
        Err(SyntheticSyntax) => (),
    }
}"#,
        );
    }

    #[test]
    fn const_pats_have_no_type_hints() {
        check_types(
            r#"
const FOO: usize = 0;

fn main() {
    match 0 {
        FOO => (),
        _ => ()
    }
}"#,
        );
    }

    #[test]
    fn let_statement() {
        check_types(
            r#"
#[derive(PartialEq)]
enum Option<T> { None, Some(T) }

#[derive(PartialEq)]
struct Test { a: Option<u32>, b: u8 }

fn main() {
    struct InnerStruct {}

    let test = 54;
      //^^^^ i32
    let test: i32 = 33;
    let mut test = 33;
          //^^^^ i32
    let _ = 22;
    let test = "test";
      //^^^^ &'static str
    let test = InnerStruct {};
      //^^^^ InnerStruct

    let test = unresolved();

    let test = (42, 'a');
      //^^^^ (i32, char)
    let (a,    (b,     (c,)) = (2, (3, (9.2,));
       //^ i32  ^ i32   ^ f64
    let &x = &92;
       //^ i32
}"#,
        );
    }

    #[test]
    fn if_expr() {
        check_types(
            r#"
//- minicore: option
struct Test { a: Option<u32>, b: u8 }

fn main() {

}"#,
        );
    }

    #[test]
    fn while_expr() {
        check_types(
            r#"
//- minicore: option
struct Test { a: Option<u32>, b: u8 }

fn main() {
    let test = Some(Test { a: Some(3), b: 1 });
      //^^^^ Option<Test>
    while let Some(Test { a: Some(x),    b: y }) = &test {};
                                //^ &u32    ^ &u8
}"#,
        );
    }

    #[test]
    fn match_arm_list() {
        check_types(
            r#"
//- minicore: option
struct Test { a: Option<u32>, b: u8 }

fn main() {
    match Some(Test { a: Some(3), b: 1 }) {
        None => (),
        test => (),
      //^^^^ Option<Test>
        Some(Test { a: Some(x), b: y }) => (),
                          //^ u32  ^ u8
        _ => {}
    }
}"#,
        );
    }

    #[test]
    fn complete_for_hint() {
        check_types(
            r#"
//- minicore: iterator
pub struct Vec<T> {}

impl<T> Vec<T> {
    pub fn new() -> Self { Vec {} }
    pub fn push(&mut self, t: T) {}
}

impl<T> IntoIterator for Vec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
}

struct IntoIter<T> {}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
}

fn main() {
    let mut data = Vec::new();
          //^^^^ Vec<&'static str>
    data.push("foo");
    for i in data {
      //^ &'static str
      let z = i;
        //^ &'static str
    }
}
"#,
        );
    }

    #[test]
    fn multi_dyn_trait_bounds() {
        check_types(
            r#"
pub struct Vec<T> {}

impl<T> Vec<T> {
    pub fn new() -> Self { Vec {} }
}

pub struct Box<T> {}

trait Display {}
auto trait Sync {}

fn main() {
    // The block expression wrapping disables the constructor hint hiding logic
    let _v = { Vec::<Box<&(dyn Display + Sync)>>::new() };
      //^^ Vec<Box<&(dyn Display + Sync)>>
    let _v = { Vec::<Box<*const (dyn Display + Sync)>>::new() };
      //^^ Vec<Box<*const (dyn Display + Sync)>>
    let _v = { Vec::<Box<dyn Display + Sync + 'static>>::new() };
      //^^ Vec<Box<dyn Display + Sync + 'static>>
}
"#,
        );
    }

    #[test]
    fn shorten_iterator_hints() {
        check_types(
            r#"
//- minicore: iterators
use core::iter;

struct MyIter;

impl Iterator for MyIter {
    type Item = ();
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() {
    let _x = MyIter;
      //^^ MyIter
    let _x = iter::repeat(0);
      //^^ impl Iterator<Item = i32>
    fn generic<T: Clone>(t: T) {
        let _x = iter::repeat(t);
          //^^ impl Iterator<Item = T>
        let _chained = iter::repeat(t).take(10);
          //^^^^^^^^ impl Iterator<Item = T>
    }
}
"#,
        );
    }

    #[test]
    fn skip_constructor_and_enum_type_hints() {
        check_with_config(
            InlayHintsConfig {
                type_hints: true,
                hide_named_constructor_hints: true,
                ..DISABLED_CONFIG
            },
            r#"
//- minicore: try, option
use core::ops::ControlFlow;

mod x {
    pub mod y { pub struct Foo; }
    pub struct Foo;
    pub enum AnotherEnum {
        Variant()
    };
}
struct Struct;
struct TupleStruct();

impl Struct {
    fn new() -> Self {
        Struct
    }
    fn try_new() -> ControlFlow<(), Self> {
        ControlFlow::Continue(Struct)
    }
}

struct Generic<T>(T);
impl Generic<i32> {
    fn new() -> Self {
        Generic(0)
    }
}

enum Enum {
    Variant(u32)
}

fn times2(value: i32) -> i32 {
    2 * value
}

fn main() {
    let enumb = Enum::Variant(0);

    let strukt = x::Foo;
    let strukt = x::y::Foo;
    let strukt = Struct;
    let strukt = Struct::new();

    let tuple_struct = TupleStruct();

    let generic0 = Generic::new();
    //  ^^^^^^^^ Generic<i32>
    let generic1 = Generic(0);
    //  ^^^^^^^^ Generic<i32>
    let generic2 = Generic::<i32>::new();
    let generic3 = <Generic<i32>>::new();
    let generic4 = Generic::<i32>(0);


    let option = Some(0);
    //  ^^^^^^ Option<i32>
    let func = times2;
    //  ^^^^ fn times2(i32) -> i32
    let closure = |x: i32| x * 2;
    //  ^^^^^^^ impl Fn(i32) -> i32
}

fn fallible() -> ControlFlow<()> {
    let strukt = Struct::try_new()?;
}
"#,
        );
    }

    #[test]
    fn shows_constructor_type_hints_when_enabled() {
        check_types(
            r#"
//- minicore: try
use core::ops::ControlFlow;

struct Struct;
struct TupleStruct();

impl Struct {
    fn new() -> Self {
        Struct
    }
    fn try_new() -> ControlFlow<(), Self> {
        ControlFlow::Continue(Struct)
    }
}

struct Generic<T>(T);
impl Generic<i32> {
    fn new() -> Self {
        Generic(0)
    }
}

fn main() {
    let strukt = Struct::new();
     // ^^^^^^ Struct
    let tuple_struct = TupleStruct();
     // ^^^^^^^^^^^^ TupleStruct
    let generic0 = Generic::new();
     // ^^^^^^^^ Generic<i32>
    let generic1 = Generic::<i32>::new();
     // ^^^^^^^^ Generic<i32>
    let generic2 = <Generic<i32>>::new();
     // ^^^^^^^^ Generic<i32>
}

fn fallible() -> ControlFlow<()> {
    let strukt = Struct::try_new()?;
     // ^^^^^^ Struct
}
"#,
        );
    }

    #[test]
    fn closure_style() {
        check_with_config(
            InlayHintsConfig { type_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: fn
fn main() {
    let x = || 2;
      //^ impl Fn() -> i32
    let y = |t: i32| x() + t;
      //^ impl Fn(i32) -> i32
    let mut t = 5;
          //^ i32
    let z = |k: i32| { t += k; };
      //^ impl FnMut(i32)
    let p = (y, z);
      //^ (impl Fn(i32) -> i32, impl FnMut(i32))
}
            "#,
        );
        check_with_config(
            InlayHintsConfig {
                type_hints: true,
                closure_style: ClosureStyle::RANotation,
                ..DISABLED_CONFIG
            },
            r#"
//- minicore: fn
fn main() {
    let x = || 2;
      //^ || -> i32
    let y = |t: i32| x() + t;
      //^ |i32| -> i32
    let mut t = 5;
          //^ i32
    let z = |k: i32| { t += k; };
      //^ |i32| -> ()
    let p = (y, z);
      //^ (|i32| -> i32, |i32| -> ())
}
            "#,
        );
        check_with_config(
            InlayHintsConfig {
                type_hints: true,
                closure_style: ClosureStyle::Hide,
                ..DISABLED_CONFIG
            },
            r#"
//- minicore: fn
fn main() {
    let x = || 2;
      //^ …
    let y = |t: i32| x() + t;
      //^ …
    let mut t = 5;
          //^ i32
    let z = |k: i32| { t += k; };
      //^ …
    let p = (y, z);
      //^ (…, …)
}
            "#,
        );
    }

    #[test]
    fn skip_closure_type_hints() {
        check_with_config(
            InlayHintsConfig {
                type_hints: true,
                hide_closure_initialization_hints: true,
                ..DISABLED_CONFIG
            },
            r#"
//- minicore: fn
fn main() {
    let multiple_2 = |x: i32| { x * 2 };

    let multiple_2 = |x: i32| x * 2;
    //  ^^^^^^^^^^ impl Fn(i32) -> i32

    let (not) = (|x: bool| { !x });
    //   ^^^ impl Fn(bool) -> bool

    let (is_zero, _b) = (|x: usize| { x == 0 }, false);
    //   ^^^^^^^ impl Fn(usize) -> bool
    //            ^^ bool

    let plus_one = |x| { x + 1 };
    //              ^ u8
    foo(plus_one);

    let add_mul = bar(|x: u8| { x + 1 });
    //  ^^^^^^^ impl FnOnce(u8) -> u8 + ?Sized

    let closure = if let Some(6) = add_mul(2).checked_sub(1) {
    //  ^^^^^^^ fn(i32) -> i32
        |x: i32| { x * 2 }
    } else {
        |x: i32| { x * 3 }
    };
}

fn foo(f: impl FnOnce(u8) -> u8) {}

fn bar(f: impl FnOnce(u8) -> u8) -> impl FnOnce(u8) -> u8 {
    move |x: u8| f(x) * 2
}
"#,
        );
    }

    #[test]
    fn skip_closure_parameter_hints() {
        check_with_config(
            InlayHintsConfig {
                type_hints: true,
                hide_closure_parameter_hints: true,
                ..DISABLED_CONFIG
            },
            r#"
//- minicore: fn
struct Foo;
impl Foo {
    fn foo(self: Self) {}
    fn bar(self: &Self) {}
}
fn main() {
    let closure = |x, y| x + y;
    //  ^^^^^^^ impl Fn(i32, i32) -> {unknown}
    closure(2, 3);
    let point = (10, 20);
    //  ^^^^^ (i32, i32)
    let (x,      y) = point;
      // ^ i32   ^ i32
    Foo::foo(Foo);
    Foo::bar(&Foo);
}
"#,
        );
    }

    #[test]
    fn hint_truncation() {
        check_with_config(
            InlayHintsConfig { max_length: Some(8), ..TEST_CONFIG },
            r#"
struct Smol<T>(T);

struct VeryLongOuterName<T>(T);

fn main() {
    let a = Smol(0u32);
      //^ Smol<u32>
    let b = VeryLongOuterName(0usize);
      //^ VeryLongOuterName<…>
    let c = Smol(Smol(0u32))
      //^ Smol<Smol<…>>
}"#,
        );
    }

    #[test]
    fn edit_for_let_stmt() {
        check_edit(
            TEST_CONFIG,
            r#"
struct S<T>(T);
fn test<F>(v: S<(S<i32>, S<()>)>, f: F) {
    let a = v;
    let S((b, c)) = v;
    let a @ S((b, c)) = v;
    let a = f;
}
"#,
            expect![[r#"
                struct S<T>(T);
                fn test<F>(v: S<(S<i32>, S<()>)>, f: F) {
                    let a: S<(S<i32>, S<()>)> = v;
                    let S((b, c)) = v;
                    let a @ S((b, c)): S<(S<i32>, S<()>)> = v;
                    let a: F = f;
                }
            "#]],
        );
    }

    #[test]
    fn edit_for_closure_param() {
        check_edit(
            TEST_CONFIG,
            r#"
fn test<T>(t: T) {
    let f = |a, b, c| {};
    let result = f(42, "", t);
}
"#,
            expect![[r#"
                fn test<T>(t: T) {
                    let f = |a: i32, b: &'static str, c: T| {};
                    let result: () = f(42, "", t);
                }
            "#]],
        );
    }

    #[test]
    fn edit_for_closure_ret() {
        check_edit(
            TEST_CONFIG,
            r#"
struct S<T>(T);
fn test() {
    let f = || { 3 };
    let f = |a: S<usize>| { S(a) };
}
"#,
            expect![[r#"
                struct S<T>(T);
                fn test() {
                    let f = || -> i32 { 3 };
                    let f = |a: S<usize>| -> S<S<usize>> { S(a) };
                }
            "#]],
        );
    }

    #[test]
    fn edit_prefixes_paths() {
        check_edit(
            TEST_CONFIG,
            r#"
pub struct S<T>(T);
mod middle {
    pub struct S<T, U>(T, U);
    pub fn make() -> S<inner::S<i64>, super::S<usize>> { loop {} }

    mod inner {
        pub struct S<T>(T);
    }

    fn test() {
        let a = make();
    }
}
"#,
            expect![[r#"
                pub struct S<T>(T);
                mod middle {
                    pub struct S<T, U>(T, U);
                    pub fn make() -> S<inner::S<i64>, super::S<usize>> { loop {} }

                    mod inner {
                        pub struct S<T>(T);
                    }

                    fn test() {
                        let a: S<inner::S<i64>, crate::S<usize>> = make();
                    }
                }
            "#]],
        );
    }

    #[test]
    fn no_edit_for_top_pat_where_type_annotation_is_invalid() {
        check_no_edit(
            TEST_CONFIG,
            r#"
fn test() {
    if let a = 42 {}
    while let a = 42 {}
    match 42 {
        a => (),
    }
}
"#,
        )
    }

    #[test]
    fn no_edit_for_opaque_type() {
        check_no_edit(
            TEST_CONFIG,
            r#"
trait Trait {}
struct S<T>(T);
fn foo() -> impl Trait {}
fn bar() -> S<impl Trait> {}
fn test() {
    let a = foo();
    let a = bar();
    let f = || { foo() };
    let f = || { bar() };
}
"#,
        );
    }

    #[test]
    fn no_edit_for_closure_return_without_body_block() {
        let config = InlayHintsConfig {
            closure_return_type_hints: ClosureReturnTypeHints::Always,
            ..TEST_CONFIG
        };
        check_edit(
            config,
            r#"
struct S<T>(T);
fn test() {
    let f = || 3;
    let f = |a: S<usize>| S(a);
}
"#,
            expect![[r#"
            struct S<T>(T);
            fn test() {
                let f = || -> i32 { 3 };
                let f = |a: S<usize>| -> S<S<usize>> { S(a) };
            }
            "#]],
        );
    }

    #[test]
    fn type_hints_async_block() {
        check_types(
            r#"
//- minicore: future
async fn main() {
    let _x = async { 8_i32 };
      //^^ impl Future<Output = i32>
}"#,
        );
    }

    #[test]
    fn type_hints_async_block_with_tail_return_exp() {
        check_types(
            r#"
//- minicore: future
async fn main() {
    let _x = async {
      //^^ impl Future<Output = i32>
        return 8_i32;
    };
}"#,
        );
    }

    #[test]
    fn works_in_included_file() {
        check_types(
            r#"
//- minicore: include
//- /main.rs
include!("foo.rs");
//- /foo.rs
fn main() {
    let _x = 42;
      //^^ i32
}"#,
        );
    }

    #[test]
    fn collapses_nested_impl_projections() {
        check_types(
            r#"
//- minicore: sized
trait T {
    type Assoc;
    fn f(self) -> Self::Assoc;
}

trait T2 {}
trait T3<T> {}

fn f(it: impl T<Assoc: T2>) {
    let l = it.f();
     // ^ impl T2
}

fn f2<G: T<Assoc: T2 + 'static>>(it: G) {
    let l = it.f();
      //^ impl T2 + 'static
}

fn f3<G: T>(it: G) where <G as T>::Assoc: T2 {
    let l = it.f();
      //^ impl T2
}

fn f4<G: T<Assoc: T2 + T3<()>>>(it: G) {
    let l = it.f();
      //^ impl T2 + T3<()>
}

fn f5<G: T<Assoc = ()>>(it: G) {
    let l = it.f();
      //^ ()
}
"#,
        );
    }

    #[test]
    fn regression_19007() {
        check_types(
            r#"
trait Foo {
    type Assoc;

    fn foo(&self) -> Self::Assoc;
}

trait Bar {
    type Target;
}

trait Baz<T> {}

struct Struct<T: Foo> {
    field: T,
}

impl<T> Struct<T>
where
    T: Foo,
    T::Assoc: Baz<<T::Assoc as Bar>::Target> + Bar,
{
    fn f(&self) {
        let x = self.field.foo();
          //^ impl Baz<<<T as Foo>::Assoc as Bar>::Target> + Bar
    }
}
"#,
        );
    }
}
