use hir::InFile;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: trait-impl-orphan
//
// Only traits defined in the current crate can be implemented for arbitrary types
pub(crate) fn trait_impl_orphan(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::TraitImplOrphan,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0117"),
        "only traits defined in the current crate can be implemented for arbitrary types"
            .to_owned(),
        InFile::new(d.file_id, d.impl_.into()),
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn simple() {
        check_diagnostics(
            r#"
//- /foo.rs crate:foo
pub trait Foo {}
//- /bar.rs crate:bar
pub struct Bar;
//- /main.rs crate:main deps:foo,bar
struct LocalType;
trait LocalTrait {}
  impl foo::Foo for bar::Bar {}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types
impl foo::Foo for LocalType {}
impl LocalTrait for bar::Bar {}
"#,
        );
    }

    #[test]
    fn generics() {
        check_diagnostics(
            r#"
//- /foo.rs crate:foo
pub trait Foo<T> {}
//- /bar.rs crate:bar
pub struct Bar<T>(T);
//- /main.rs crate:main deps:foo,bar
struct LocalType<T>;
trait LocalTrait<T> {}
  impl<T> foo::Foo<T> for bar::Bar<T> {}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types

  impl<T> foo::Foo<T> for bar::Bar<LocalType<T>> {}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types

  impl<T> foo::Foo<LocalType<T>> for bar::Bar<T> {}

  impl<T> foo::Foo<bar::Bar<LocalType<T>>> for bar::Bar<LocalType<T>> {}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types
"#,
        );
    }

    #[test]
    fn fundamental() {
        check_diagnostics(
            r#"
//- /foo.rs crate:foo
pub trait Foo<T> {}
//- /bar.rs crate:bar
pub struct Bar<T>(T);
#[lang = "owned_box"]
#[fundamental]
pub struct Box<T>(T);
//- /main.rs crate:main deps:foo,bar
struct LocalType;
  impl<T> foo::Foo<T> for bar::Box<T> {}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types
  impl<T> foo::Foo<T> for &LocalType {}
  impl<T> foo::Foo<T> for bar::Box<LocalType> {}
"#,
        );
    }

    #[test]
    fn dyn_object() {
        check_diagnostics(
            r#"
//- /foo.rs crate:foo
pub trait Foo<T> {}
//- /bar.rs crate:bar
pub struct Bar;
//- /main.rs crate:main deps:foo,bar
trait LocalTrait {}
impl<T> foo::Foo<T> for dyn LocalTrait {}
impl<T> foo::Foo<dyn LocalTrait> for Bar {}
"#,
        );
    }

    #[test]
    fn twice_fundamental() {
        check_diagnostics(
            r#"
//- /foo.rs crate:foo
pub trait Trait {}
//- /bar.rs crate:bar deps:foo
struct Foo;
impl foo::Trait for &&Foo {}
        "#,
        );
    }
}
