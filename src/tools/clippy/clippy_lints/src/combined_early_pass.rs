//! A statically-combined early lint pass.
//!
//! The early-pass analogue of [`combined_late_pass`]. Folds clippy's early
//! passes into one concrete struct, one field per pass, with a single
//! `EarlyLintPass` impl that forwards each `check_*` to every field. Same
//! static-dispatch / DCE win as the late version: because the field types are
//! concrete and the forwards are `#[inline(always)]`, a pass that doesn't
//! override a `check_*` contributes only the empty default body, which is
//! DCE'd away. So the per-node, per-pass indirect (vtable) call into an empty
//! method disappears entirely, and the passes that do override become direct,
//! inlined calls. No vtable, no per-node dynamic dispatch.
//!
//! Unlike the late combine there is no `active` gate. rustc drops fully-disabled
//! late passes via `skippable_lints`, but the early pass runner has
//! no such filtering, so a plain forward is equivalent and loses nothing.
//!
//! [`combined_late_pass`]: crate::combined_late_pass

/// Run one field's `check_*`.
///
/// Fully qualified through [`rustc_lint::EarlyLintPass`] since some passes impl
/// both `EarlyLintPass` and `LateLintPass` with like-named methods, which would
/// be ambiguous on the concrete field type.
#[macro_export]
macro_rules! run_combined_early_lint_pass_field {
    ($self:ident, $field:ident, $name:ident, ($($arg:expr),* $(,)?)) => {
        rustc_lint::EarlyLintPass::$name(&mut $self.$field, $($arg),*);
    };
}

/// Forward one `check_*` method to every field of the combined pass.
#[macro_export]
macro_rules! expand_combined_early_lint_pass_method {
    ([$($field:ident),*], $self:ident, $name:ident, $args:tt) => ({
        $($crate::run_combined_early_lint_pass_field!($self, $field, $name, $args);)*
    })
}

/// Generate the combined `EarlyLintPass` impl's `check_*` methods, one per method
/// in rustc's early-pass method list.
#[macro_export]
macro_rules! expand_combined_early_lint_pass_methods {
    ($fields:tt, [$($(#[$attr:meta])* fn $name:ident($($param:ident: $arg:ty),*);)*]) => (
        $(#[inline(always)] fn $name(&mut self, cx: &rustc_lint::EarlyContext<'_>, $($param: $arg),*) {
            $crate::expand_combined_early_lint_pass_method!($fields, self, $name, (cx, $($param),*));
        })*
    )
}

/// Declare the combined struct (one field per pass) plus its
/// `LintPass`/`EarlyLintPass` impls. The method list comes from
/// `rustc_lint::early_lint_methods!` so it can't drift from rustc's.
///
/// Each entry is `Field: Type = constructor`; `new`'s params (`conf`, ...) come
/// from the caller so ctor exprs can name them without hygiene trouble.
#[macro_export]
macro_rules! combined_early_lint_pass {
    (
        [$name:ident, ($($pname:ident: $pty:ty),* $(,)?), [$($field:ident: $fty:ty = $ctor:expr,)*]],
        $methods:tt
    ) => {
        #[allow(non_snake_case)]
        pub struct $name {
            $($field: $fty,)*
        }

        impl $name {
            pub fn new($($pname: $pty,)*) -> Self {
                Self {
                    $($field: $ctor,)*
                }
            }
        }

        #[allow(rustc::lint_pass_impl_without_macro)]
        impl rustc_lint::LintPass for $name {
            fn name(&self) -> &'static str {
                stringify!($name)
            }
            fn get_lints(&self) -> rustc_lint::LintVec {
                // Reserve at least one slot per pass up front to skip the early reallocations.
                let mut lints = Vec::with_capacity([$(stringify!($field)),*].len());
                $(lints.extend(self.$field.get_lints());)*
                lints
            }
        }

        impl rustc_lint::EarlyLintPass for $name {
            $crate::expand_combined_early_lint_pass_methods!([$($field),*], $methods);
        }
    };
}
