use either::Either;
use hir::GenericArgsProhibitedReason;
use ide_db::assists::Assist;
use ide_db::source_change::SourceChange;
use ide_db::text_edit::TextEdit;
use syntax::{AstNode, TextRange, ast};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, fix};

// Diagnostic: generic-args-prohibited
//
// This diagnostic is shown when generic arguments are provided for a type that does not accept
// generic arguments.
pub(crate) fn generic_args_prohibited(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::GenericArgsProhibited,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0109"),
        describe_reason(d.reason),
        d.args.map(Into::into),
    )
    .stable()
    .with_fixes(fixes(ctx, d))
}

fn describe_reason(reason: GenericArgsProhibitedReason) -> String {
    let kind = match reason {
        GenericArgsProhibitedReason::Module => "modules",
        GenericArgsProhibitedReason::TyParam => "type parameters",
        GenericArgsProhibitedReason::SelfTy => "`Self`",
        GenericArgsProhibitedReason::PrimitiveTy => "builtin types",
        GenericArgsProhibitedReason::EnumVariant => {
            return "you can specify generic arguments on either the enum or the variant, but not both"
                .to_owned();
        }
        GenericArgsProhibitedReason::Const => "constants",
        GenericArgsProhibitedReason::Static => "statics",
        GenericArgsProhibitedReason::LocalVariable => "local variables",
    };
    format!("generic arguments are not allowed on {kind}")
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::GenericArgsProhibited) -> Option<Vec<Assist>> {
    let file_id = d.args.file_id.file_id()?;
    let syntax = d.args.to_node(ctx.sema.db);
    let range = match &syntax {
        Either::Left(_) => syntax.syntax().text_range(),
        Either::Right(param_list) => {
            let path_segment = ast::PathSegment::cast(param_list.syntax().parent()?)?;
            let start = if let Some(coloncolon) = path_segment.coloncolon_token() {
                coloncolon.text_range().start()
            } else {
                param_list.syntax().text_range().start()
            };
            let end = if let Some(ret_type) = path_segment.ret_type() {
                ret_type.syntax().text_range().end()
            } else {
                param_list.syntax().text_range().end()
            };
            TextRange::new(start, end)
        }
    };
    Some(vec![fix(
        "remove_generic_args",
        "Remove these generics",
        SourceChange::from_text_edit(file_id.file_id(ctx.sema.db), TextEdit::delete(range)),
        syntax.syntax().text_range(),
    )])
}

#[cfg(test)]
mod tests {
    // This diagnostic was the first to be emitted in ty lowering, so the tests here also test
    // diagnostics in ty lowering in general (which is why there are so many of them).

    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn primitives() {
        check_diagnostics(
            r#"
//- /core.rs crate:core library
#![rustc_coherence_is_core]
impl str {
    pub fn trim() {}
}

//- /lib.rs crate:foo deps:core
fn bar<T>() {}

fn foo() {
    let _: (bool<()>, ());
             // ^^^^ 💡 error: generic arguments are not allowed on builtin types
    let _ = <str<'_>>::trim;
             // ^^^^ 💡 error: generic arguments are not allowed on builtin types
    bar::<u32<{ const { 1 + 1 } }>>();
          // ^^^^^^^^^^^^^^^^^^^^^ 💡 error: generic arguments are not allowed on builtin types
}
            "#,
        );
    }

    #[test]
    fn modules() {
        check_diagnostics(
            r#"
pub mod foo {
    pub mod bar {
        pub struct Baz;

        impl Baz {
            pub fn qux() {}
        }
    }
}

fn foo() {
    let _: foo::<'_>::bar::Baz;
           // ^^^^^^ 💡 error: generic arguments are not allowed on modules
    let _ = <foo::bar<()>::Baz>::qux;
                  // ^^^^ 💡 error: generic arguments are not allowed on modules
}
        "#,
        );
    }

    #[test]
    fn type_parameters() {
        check_diagnostics(
            r#"
fn foo<T, U>() {
    let _: T<'a>;
         // ^^^^ 💡 error: generic arguments are not allowed on type parameters
    let _: U::<{ 1 + 2 }>;
         // ^^^^^^^^^^^^^ 💡 error: generic arguments are not allowed on type parameters
}
        "#,
        );
    }

    #[test]
    fn fn_like_generic_args() {
        check_diagnostics(
            r#"
fn foo() {
    let _: bool(bool, i32) -> ();
            // ^^^^^^^^^^^ 💡 error: generic arguments are not allowed on builtin types
}
        "#,
        );
    }

    #[test]
    fn fn_signature() {
        check_diagnostics(
            r#"
fn foo(
    _a: bool<'_>,
         // ^^^^ 💡 error: generic arguments are not allowed on builtin types
    _b: i32::<i64>,
        // ^^^^^^^ 💡 error: generic arguments are not allowed on builtin types
    _c: &(&str<1>)
           // ^^^ 💡 error: generic arguments are not allowed on builtin types
) -> ((), i32<bool>) {
          // ^^^^^^ 💡 error: generic arguments are not allowed on builtin types
    ((), 0)
}
        "#,
        );
    }

    #[test]
    fn const_static_type() {
        check_diagnostics(
            r#"
const A: i32<bool> = 0;
         // ^^^^^^ 💡 error: generic arguments are not allowed on builtin types
static A: i32::<{ 1 + 3 }> = 0;
          // ^^^^^^^^^^^^^ 💡 error: generic arguments are not allowed on builtin types
        "#,
        );
    }

    #[test]
    fn fix() {
        check_fix(
            r#"
fn foo() {
    let _: bool<'_, (), { 1 + 1 }>$0;
}"#,
            r#"
fn foo() {
    let _: bool;
}"#,
        );
        check_fix(
            r#"
fn foo() {
    let _: bool::$0<'_, (), { 1 + 1 }>;
}"#,
            r#"
fn foo() {
    let _: bool;
}"#,
        );
        check_fix(
            r#"
fn foo() {
    let _: bool(i$032);
}"#,
            r#"
fn foo() {
    let _: bool;
}"#,
        );
        check_fix(
            r#"
fn foo() {
    let _: bool$0(i32) -> i64;
}"#,
            r#"
fn foo() {
    let _: bool;
}"#,
        );
        check_fix(
            r#"
fn foo() {
    let _: bool::(i$032) -> i64;
}"#,
            r#"
fn foo() {
    let _: bool;
}"#,
        );
        check_fix(
            r#"
fn foo() {
    let _: bool::(i32)$0;
}"#,
            r#"
fn foo() {
    let _: bool;
}"#,
        );
    }

    #[test]
    fn in_fields() {
        check_diagnostics(
            r#"
struct A(bool<i32>);
          // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
struct B { v: bool<(), 1> }
               // ^^^^^^^ 💡 error: generic arguments are not allowed on builtin types
union C {
    a: bool<i32>,
        // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
    b: i32<bool>,
       // ^^^^^^ 💡 error: generic arguments are not allowed on builtin types
       }
enum D {
    A(bool<i32>),
       // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
    B { v: i32<bool> },
           // ^^^^^^ 💡 error: generic arguments are not allowed on builtin types
}
        "#,
        );
    }

    #[test]
    fn in_generics() {
        check_diagnostics(
            r#"
mod foo {
    pub trait Trait {}
}

struct A<A: foo::<()>::Trait>(A)
            // ^^^^^^ 💡 error: generic arguments are not allowed on modules
    where bool<i32>: foo::Trait;
           // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
union B<A: foo::<()>::Trait>
           // ^^^^^^ 💡 error: generic arguments are not allowed on modules
    where bool<i32>: foo::Trait
           // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
{ a: A }
enum C<A: foo::<()>::Trait>
          // ^^^^^^ 💡 error: generic arguments are not allowed on modules
    where bool<i32>: foo::Trait
           // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
{}

fn f<A: foo::<()>::Trait>()
        // ^^^^^^ 💡 error: generic arguments are not allowed on modules
    where bool<i32>: foo::Trait
           // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
{}

type D<A: foo::<()>::Trait> = A
          // ^^^^^^ 💡 error: generic arguments are not allowed on modules
    where bool<i32>: foo::Trait;
           // ^^^^^ 💡 error: generic arguments are not allowed on builtin types

trait E<A: foo::<()>::Trait>
           // ^^^^^^ 💡 error: generic arguments are not allowed on modules
    where bool<i32>: foo::Trait
           // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
{
    fn f<B: foo::<()>::Trait>()
            // ^^^^^^ 💡 error: generic arguments are not allowed on modules
        where bool<i32>: foo::Trait
               // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
    {}

    type D<B: foo::<()>::Trait> = A
              // ^^^^^^ 💡 error: generic arguments are not allowed on modules
        where bool<i32>: foo::Trait;
               // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
}

impl<A: foo::<()>::Trait> E<()> for ()
        // ^^^^^^ 💡 error: generic arguments are not allowed on modules
    where bool<i32>: foo::Trait
           // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
{
    fn f<B: foo::<()>::Trait>()
            // ^^^^^^ 💡 error: generic arguments are not allowed on modules
        where bool<i32>: foo::Trait
               // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
    {}

    type D<B: foo::<()>::Trait> = A
              // ^^^^^^ 💡 error: generic arguments are not allowed on modules
        where bool<i32>: foo::Trait;
               // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
}
        "#,
        );
    }

    #[test]
    fn assoc_items() {
        check_diagnostics(
            r#"
struct Foo;

trait Trait {
    fn f() -> bool<i32> { true }
               // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
    type T = i32<bool>;
             // ^^^^^^ 💡 error: generic arguments are not allowed on builtin types
}

impl Trait for Foo {
    fn f() -> bool<i32> { true }
               // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
    type T = i32<bool>;
             // ^^^^^^ 💡 error: generic arguments are not allowed on builtin types
}

impl Foo {
    fn f() -> bool<i32> { true }
               // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
    type T = i32<bool>;
             // ^^^^^^ 💡 error: generic arguments are not allowed on builtin types
}
        "#,
        );
    }

    #[test]
    fn const_param_ty() {
        check_diagnostics(
            r#"
fn foo<
    const A: bool<i32>,
              // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
    B,
    C,
    const D: bool<i32>,
              // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
    const E: bool<i32>,
              // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
>() {}
        "#,
        );
    }

    #[test]
    fn generic_defaults() {
        check_diagnostics(
            r#"
struct Foo<A = bool<i32>>(A);
                // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
        "#,
        );
    }

    #[test]
    fn impl_self_ty() {
        check_diagnostics(
            r#"
struct Foo<A>(A);
trait Trait {}
impl Foo<bool<i32>> {}
          // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
impl Trait for Foo<bool<i32>> {}
                    // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
        "#,
        );
    }

    #[test]
    fn impl_trait() {
        check_diagnostics(
            r#"
mod foo {
    pub trait Trait {}
}
impl foo::<()>::Trait for () {}
     // ^^^^^^ 💡 error: generic arguments are not allowed on modules
        "#,
        );
    }

    #[test]
    fn type_alias() {
        check_diagnostics(
            r#"
pub trait Trait {
    type Assoc;
}
type T = bool<i32>;
          // ^^^^^ 💡 error: generic arguments are not allowed on builtin types
impl Trait for () {
    type Assoc = i32<bool>;
                 // ^^^^^^ 💡 error: generic arguments are not allowed on builtin types
}
        "#,
        );
    }

    #[test]
    fn in_record_expr() {
        check_diagnostics(
            r#"
mod foo {
    pub struct Bar { pub field: i32 }
}
fn baz() {
    let _ = foo::<()>::Bar { field: 0 };
            // ^^^^^^ 💡 error: generic arguments are not allowed on modules
}
        "#,
        );
    }

    #[test]
    fn in_record_pat() {
        check_diagnostics(
            r#"
mod foo {
    pub struct Bar { field: i32 }
}
fn baz(v: foo::Bar) {
    let foo::<()>::Bar { .. } = v;
        // ^^^^^^ 💡 error: generic arguments are not allowed on modules
}
        "#,
        );
    }

    #[test]
    fn in_tuple_struct_pat() {
        check_diagnostics(
            r#"
mod foo {
    pub struct Bar(i32);
}
fn baz(v: foo::Bar) {
    let foo::<()>::Bar(..) = v;
        // ^^^^^^ 💡 error: generic arguments are not allowed on modules
}
        "#,
        );
    }

    #[test]
    fn in_path_pat() {
        check_diagnostics(
            r#"
mod foo {
    pub struct Bar;
}
fn baz(v: foo::Bar) {
    let foo::<()>::Bar = v;
        // ^^^^^^ 💡 error: generic arguments are not allowed on modules
}
        "#,
        );
    }

    #[test]
    fn in_path_expr() {
        check_diagnostics(
            r#"
mod foo {
    pub struct Bar;
}
fn baz() {
    let _ = foo::<()>::Bar;
            // ^^^^^^ 💡 error: generic arguments are not allowed on modules
}
        "#,
        );
    }

    #[test]
    fn const_param_and_static() {
        check_diagnostics(
            r#"
const CONST: i32 = 0;
static STATIC: i32 = 0;
fn baz<const CONST_PARAM: usize>() {
    let _ = CONST_PARAM::<()>;
                    // ^^^^^^ 💡 error: generic arguments are not allowed on constants
    let _ = STATIC::<()>;
               // ^^^^^^ 💡 error: generic arguments are not allowed on statics
}
        "#,
        );
    }

    #[test]
    fn local_variable() {
        check_diagnostics(
            r#"
fn baz() {
    let x = 1;
    let _ = x::<()>;
          // ^^^^^^ 💡 error: generic arguments are not allowed on local variables
}
        "#,
        );
    }

    #[test]
    fn enum_variant() {
        check_diagnostics(
            r#"
enum Enum<A> {
    Variant(A),
}
mod enum_ {
    pub(super) use super::Enum::Variant as V;
}
fn baz() {
    let v = Enum::<()>::Variant::<()>(());
                            // ^^^^^^ 💡 error: you can specify generic arguments on either the enum or the variant, but not both
    let Enum::<()>::Variant::<()>(..) = v;
                        // ^^^^^^ 💡 error: you can specify generic arguments on either the enum or the variant, but not both
    let _ = Enum::<()>::Variant(());
    let _ = Enum::Variant::<()>(());
}
fn foo() {
    use Enum::Variant;
    let _ = Variant::<()>(());
    let _ = enum_::V::<()>(());
    let _ = enum_::<()>::V::<()>(());
              // ^^^^^^ 💡 error: generic arguments are not allowed on modules
}
        "#,
        );
    }

    #[test]
    fn dyn_trait() {
        check_diagnostics(
            r#"
mod foo {
    pub trait Trait {}
}

fn bar() {
    let _: &dyn foo::<()>::Trait;
                // ^^^^^^ 💡 error: generic arguments are not allowed on modules
    let _: &foo::<()>::Trait;
            // ^^^^^^ 💡 error: generic arguments are not allowed on modules
}
        "#,
        );
    }

    #[test]
    fn regression_18768() {
        check_diagnostics(
            r#"
//- minicore: result
//- /foo.rs crate:foo edition:2018
pub mod lib {
    mod core {
        pub use core::*;
    }
    pub use self::core::result;
}

pub mod __private {
    pub use crate::lib::result::Result::{self, Err, Ok};
}

//- /bar.rs crate:bar deps:foo edition:2018
fn bar() {
    _ = foo::__private::Result::<(), ()>::Ok;
}
        "#,
        );
    }

    #[test]
    fn enum_variant_type_ns() {
        check_diagnostics(
            r#"
enum KvnDeserializerErr<I> {
    UnexpectedKeyword { found: I, expected: I },
}

fn foo() {
    let _x: KvnDeserializerErr<()> =
        KvnDeserializerErr::<()>::UnexpectedKeyword { found: (), expected: () };
    let _x: KvnDeserializerErr<()> =
        KvnDeserializerErr::<()>::UnexpectedKeyword::<()> { found: (), expected: () };
                                                // ^^^^^^ 💡 error: you can specify generic arguments on either the enum or the variant, but not both
}
        "#,
        );
    }
}
