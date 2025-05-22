use hir::InFile;
use syntax::ast;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, Severity, adjusted_display_range};

// Diagnostic: trait-impl-incorrect-safety
//
// Diagnoses incorrect safety annotations of trait impls.
pub(crate) fn trait_impl_incorrect_safety(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::TraitImplIncorrectSafety,
) -> Diagnostic {
    Diagnostic::new(
        DiagnosticCode::Ra("trait-impl-incorrect-safety", Severity::Error),
        if d.should_be_safe {
            "unsafe impl for safe trait"
        } else {
            "impl for unsafe trait needs to be unsafe"
        },
        adjusted_display_range::<ast::Impl>(
            ctx,
            InFile { file_id: d.file_id, value: d.impl_ },
            &|impl_| {
                if d.should_be_safe {
                    Some(match (impl_.unsafe_token(), impl_.impl_token()) {
                        (None, None) => return None,
                        (None, Some(t)) | (Some(t), None) => t.text_range(),
                        (Some(t1), Some(t2)) => t1.text_range().cover(t2.text_range()),
                    })
                } else {
                    impl_.impl_token().map(|t| t.text_range())
                }
            },
        ),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn simple() {
        check_diagnostics(
            r#"
trait Safe {}
unsafe trait Unsafe {}

  impl Safe for () {}

  impl Unsafe for () {}
//^^^^  error: impl for unsafe trait needs to be unsafe

  unsafe impl Safe for () {}
//^^^^^^^^^^^ error: unsafe impl for safe trait

  unsafe impl Unsafe for () {}
"#,
        );
    }

    #[test]
    fn drop_may_dangle() {
        check_diagnostics(
            r#"
#[lang = "drop"]
trait Drop {}
struct S<T>;
struct L<'l>;

  impl<T> Drop for S<T> {}

  impl<#[may_dangle] T> Drop for S<T> {}
//^^^^ error: impl for unsafe trait needs to be unsafe

  unsafe impl<T> Drop for S<T> {}
//^^^^^^^^^^^ error: unsafe impl for safe trait

  unsafe impl<#[may_dangle] T> Drop for S<T> {}

  impl<'l> Drop for L<'l> {}

  impl<#[may_dangle] 'l> Drop for L<'l> {}
//^^^^ error: impl for unsafe trait needs to be unsafe

  unsafe impl<'l> Drop for L<'l> {}
//^^^^^^^^^^^ error: unsafe impl for safe trait

  unsafe impl<#[may_dangle] 'l> Drop for L<'l> {}
"#,
        );
    }

    #[test]
    fn negative() {
        check_diagnostics(
            r#"
trait Trait {}

  impl !Trait for () {}

  unsafe impl !Trait for () {}
//^^^^^^^^^^^ error: unsafe impl for safe trait

unsafe trait UnsafeTrait {}

  impl !UnsafeTrait for () {}

  unsafe impl !UnsafeTrait for () {}
//^^^^^^^^^^^ error: unsafe impl for safe trait

"#,
        );
    }

    #[test]
    fn inherent() {
        check_diagnostics(
            r#"
struct S;

  impl S {}

  unsafe impl S {}
//^^^^^^^^^^^ error: unsafe impl for safe trait
"#,
        );
    }
}
