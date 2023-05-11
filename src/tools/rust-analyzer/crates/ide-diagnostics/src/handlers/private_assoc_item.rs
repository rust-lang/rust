use either::Either;

use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: private-assoc-item
//
// This diagnostic is triggered if the referenced associated item is not visible from the current
// module.
pub(crate) fn private_assoc_item(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::PrivateAssocItem,
) -> Diagnostic {
    // FIXME: add quickfix
    let name = d.item.name(ctx.sema.db).map(|name| format!("`{name}` ")).unwrap_or_default();
    Diagnostic::new(
        "private-assoc-item",
        format!(
            "{} {}is private",
            match d.item {
                hir::AssocItem::Function(_) => "function",
                hir::AssocItem::Const(_) => "const",
                hir::AssocItem::TypeAlias(_) => "type alias",
            },
            name,
        ),
        ctx.sema
            .diagnostics_display_range(d.expr_or_pat.clone().map(|it| match it {
                Either::Left(it) => it.into(),
                Either::Right(it) => match it {
                    Either::Left(it) => it.into(),
                    Either::Right(it) => it.into(),
                },
            }))
            .range,
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn private_method() {
        check_diagnostics(
            r#"
mod module {
    pub struct Struct;
    impl Struct {
        fn method(&self) {}
    }
}
fn main(s: module::Struct) {
    s.method();
  //^^^^^^^^^^ error: function `method` is private
}
"#,
        );
    }

    #[test]
    fn private_func() {
        check_diagnostics(
            r#"
mod module {
    pub struct Struct;
    impl Struct {
        fn func() {}
    }
}
fn main() {
    module::Struct::func();
  //^^^^^^^^^^^^^^^^^^^^ error: function `func` is private
}
"#,
        );
    }

    #[test]
    fn private_const() {
        check_diagnostics(
            r#"
mod module {
    pub struct Struct;
    impl Struct {
        const CONST: u32 = 0;
    }
}
fn main() {
    module::Struct::CONST;
  //^^^^^^^^^^^^^^^^^^^^^ error: const `CONST` is private
}
"#,
        );
    }

    #[test]
    fn private_but_shadowed_in_deref() {
        check_diagnostics(
            r#"
//- minicore: deref
mod module {
    pub struct Struct { field: Inner }
    pub struct Inner;
    impl core::ops::Deref for Struct {
        type Target = Inner;
        fn deref(&self) -> &Inner { &self.field }
    }
    impl Struct {
        fn method(&self) {}
    }
    impl Inner {
        pub fn method(&self) {}
    }
}
fn main(s: module::Struct) {
    s.method();
}
"#,
        );
    }

    #[test]
    fn can_see_through_top_level_anonymous_const() {
        // regression test for #14046.
        check_diagnostics(
            r#"
struct S;
mod m {
    const _: () = {
        impl crate::S {
            pub(crate) fn method(self) {}
            pub(crate) const A: usize = 42;
        }
    };
    mod inner {
        const _: () = {
            impl crate::S {
                pub(crate) fn method2(self) {}
                pub(crate) const B: usize = 42;
                pub(super) fn private(self) {}
                pub(super) const PRIVATE: usize = 42;
            }
        };
    }
}
fn main() {
    S.method();
    S::A;
    S.method2();
    S::B;
    S.private();
  //^^^^^^^^^^^ error: function `private` is private
    S::PRIVATE;
  //^^^^^^^^^^ error: const `PRIVATE` is private
}
"#,
        );
    }
}
