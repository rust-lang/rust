use crate::context::{EarlyContext, LateContext};

use rustc_ast as ast;
use rustc_hir as hir;
use rustc_session::lint::builtin::HardwiredLints;
use rustc_session::lint::LintPass;
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::Ident;
use rustc_span::Span;

#[macro_export]
macro_rules! late_lint_methods {
    ($macro:path, $args:tt) => (
        $macro!($args, [
            fn check_body(a: &'tcx hir::Body<'tcx>);
            fn check_body_post(a: &'tcx hir::Body<'tcx>);
            fn check_crate();
            fn check_crate_post();
            fn check_mod(a: &'tcx hir::Mod<'tcx>, b: hir::HirId);
            fn check_foreign_item(a: &'tcx hir::ForeignItem<'tcx>);
            fn check_item(a: &'tcx hir::Item<'tcx>);
            fn check_item_post(a: &'tcx hir::Item<'tcx>);
            fn check_local(a: &'tcx hir::Local<'tcx>);
            fn check_block(a: &'tcx hir::Block<'tcx>);
            fn check_block_post(a: &'tcx hir::Block<'tcx>);
            fn check_stmt(a: &'tcx hir::Stmt<'tcx>);
            fn check_arm(a: &'tcx hir::Arm<'tcx>);
            fn check_pat(a: &'tcx hir::Pat<'tcx>);
            fn check_expr(a: &'tcx hir::Expr<'tcx>);
            fn check_expr_post(a: &'tcx hir::Expr<'tcx>);
            fn check_ty(a: &'tcx hir::Ty<'tcx>);
            fn check_generic_param(a: &'tcx hir::GenericParam<'tcx>);
            fn check_generics(a: &'tcx hir::Generics<'tcx>);
            fn check_poly_trait_ref(a: &'tcx hir::PolyTraitRef<'tcx>);
            fn check_fn(
                a: rustc_hir::intravisit::FnKind<'tcx>,
                b: &'tcx hir::FnDecl<'tcx>,
                c: &'tcx hir::Body<'tcx>,
                d: Span,
                e: LocalDefId);
            fn check_trait_item(a: &'tcx hir::TraitItem<'tcx>);
            fn check_impl_item(a: &'tcx hir::ImplItem<'tcx>);
            fn check_impl_item_post(a: &'tcx hir::ImplItem<'tcx>);
            fn check_struct_def(a: &'tcx hir::VariantData<'tcx>);
            fn check_field_def(a: &'tcx hir::FieldDef<'tcx>);
            fn check_variant(a: &'tcx hir::Variant<'tcx>);
            fn check_path(a: &hir::Path<'tcx>, b: hir::HirId);
            fn check_attribute(a: &'tcx ast::Attribute);

            /// Called when entering a syntax node that can have lint attributes such
            /// as `#[allow(...)]`. Called with *all* the attributes of that node.
            fn enter_lint_attrs(a: &'tcx [ast::Attribute]);

            /// Counterpart to `enter_lint_attrs`.
            fn exit_lint_attrs(a: &'tcx [ast::Attribute]);
        ]);
    )
}

/// Trait for types providing lint checks.
///
/// Each `check` method checks a single syntax node, and should not
/// invoke methods recursively (unlike `Visitor`). By default they
/// do nothing.
//
// FIXME: eliminate the duplication with `Visitor`. But this also
// contains a few lint-specific methods with no equivalent in `Visitor`.

macro_rules! declare_late_lint_pass {
    ([], [$($(#[$attr:meta])* fn $name:ident($($param:ident: $arg:ty),*);)*]) => (
        pub trait LateLintPass<'tcx>: LintPass {
            $(#[inline(always)] fn $name(&mut self, _: &LateContext<'tcx>, $(_: $arg),*) {})*
        }
    )
}

// Declare the `LateLintPass` trait, which contains empty default definitions
// for all the `check_*` methods.
late_lint_methods!(declare_late_lint_pass, []);

impl LateLintPass<'_> for HardwiredLints {}

#[macro_export]
macro_rules! expand_combined_late_lint_pass_method {
    ([$($pass:ident),*], $self: ident, $name: ident, $params:tt) => ({
        $($self.$pass.$name $params;)*
    })
}

#[macro_export]
macro_rules! expand_combined_late_lint_pass_methods {
    ($passes:tt, [$($(#[$attr:meta])* fn $name:ident($($param:ident: $arg:ty),*);)*]) => (
        $(fn $name(&mut self, context: &LateContext<'tcx>, $($param: $arg),*) {
            expand_combined_late_lint_pass_method!($passes, self, $name, (context, $($param),*));
        })*
    )
}

/// Combines multiple lints passes into a single lint pass, at compile time,
/// for maximum speed. Each `check_foo` method in `$methods` within this pass
/// simply calls `check_foo` once per `$pass`. Compare with
/// `LateLintPassObjects`, which is similar, but combines lint passes at
/// runtime.
#[macro_export]
macro_rules! declare_combined_late_lint_pass {
    ([$v:vis $name:ident, [$($pass:ident: $constructor:expr,)*]], $methods:tt) => (
        #[allow(non_snake_case)]
        $v struct $name {
            $($pass: $pass,)*
        }

        impl $name {
            $v fn new() -> Self {
                Self {
                    $($pass: $constructor,)*
                }
            }

            $v fn get_lints() -> LintArray {
                let mut lints = Vec::new();
                $(lints.extend_from_slice(&$pass::get_lints());)*
                lints
            }
        }

        impl<'tcx> LateLintPass<'tcx> for $name {
            expand_combined_late_lint_pass_methods!([$($pass),*], $methods);
        }

        #[allow(rustc::lint_pass_impl_without_macro)]
        impl LintPass for $name {
            fn name(&self) -> &'static str {
                panic!()
            }
        }
    )
}

#[macro_export]
macro_rules! early_lint_methods {
    ($macro:path, $args:tt) => (
        $macro!($args, [
            fn check_param(a: &ast::Param);
            fn check_ident(a: Ident);
            fn check_crate(a: &ast::Crate);
            fn check_crate_post(a: &ast::Crate);
            fn check_item(a: &ast::Item);
            fn check_item_post(a: &ast::Item);
            fn check_local(a: &ast::Local);
            fn check_block(a: &ast::Block);
            fn check_stmt(a: &ast::Stmt);
            fn check_arm(a: &ast::Arm);
            fn check_pat(a: &ast::Pat);
            fn check_pat_post(a: &ast::Pat);
            fn check_expr(a: &ast::Expr);
            fn check_ty(a: &ast::Ty);
            fn check_generic_arg(a: &ast::GenericArg);
            fn check_generic_param(a: &ast::GenericParam);
            fn check_generics(a: &ast::Generics);
            fn check_poly_trait_ref(a: &ast::PolyTraitRef);
            fn check_fn(a: rustc_ast::visit::FnKind<'_>, c: Span, d_: ast::NodeId);
            fn check_trait_item(a: &ast::AssocItem);
            fn check_impl_item(a: &ast::AssocItem);
            fn check_variant(a: &ast::Variant);
            fn check_attribute(a: &ast::Attribute);
            fn check_mac_def(a: &ast::MacroDef);
            fn check_mac(a: &ast::MacCall);

            /// Called when entering a syntax node that can have lint attributes such
            /// as `#[allow(...)]`. Called with *all* the attributes of that node.
            fn enter_lint_attrs(a: &[ast::Attribute]);

            /// Counterpart to `enter_lint_attrs`.
            fn exit_lint_attrs(a: &[ast::Attribute]);

            fn enter_where_predicate(a: &ast::WherePredicate);
            fn exit_where_predicate(a: &ast::WherePredicate);
        ]);
    )
}

macro_rules! declare_early_lint_pass {
    ([], [$($(#[$attr:meta])* fn $name:ident($($param:ident: $arg:ty),*);)*]) => (
        pub trait EarlyLintPass: LintPass {
            $(#[inline(always)] fn $name(&mut self, _: &EarlyContext<'_>, $(_: $arg),*) {})*
        }
    )
}

// Declare the `EarlyLintPass` trait, which contains empty default definitions
// for all the `check_*` methods.
early_lint_methods!(declare_early_lint_pass, []);

#[macro_export]
macro_rules! expand_combined_early_lint_pass_method {
    ([$($pass:ident),*], $self: ident, $name: ident, $params:tt) => ({
        $($self.$pass.$name $params;)*
    })
}

#[macro_export]
macro_rules! expand_combined_early_lint_pass_methods {
    ($passes:tt, [$($(#[$attr:meta])* fn $name:ident($($param:ident: $arg:ty),*);)*]) => (
        $(fn $name(&mut self, context: &EarlyContext<'_>, $($param: $arg),*) {
            expand_combined_early_lint_pass_method!($passes, self, $name, (context, $($param),*));
        })*
    )
}

/// Combines multiple lints passes into a single lint pass, at compile time,
/// for maximum speed. Each `check_foo` method in `$methods` within this pass
/// simply calls `check_foo` once per `$pass`. Compare with
/// `EarlyLintPassObjects`, which is similar, but combines lint passes at
/// runtime.
#[macro_export]
macro_rules! declare_combined_early_lint_pass {
    ([$v:vis $name:ident, [$($pass:ident: $constructor:expr,)*]], $methods:tt) => (
        #[allow(non_snake_case)]
        $v struct $name {
            $($pass: $pass,)*
        }

        impl $name {
            $v fn new() -> Self {
                Self {
                    $($pass: $constructor,)*
                }
            }

            $v fn get_lints() -> LintArray {
                let mut lints = Vec::new();
                $(lints.extend_from_slice(&$pass::get_lints());)*
                lints
            }
        }

        impl EarlyLintPass for $name {
            expand_combined_early_lint_pass_methods!([$($pass),*], $methods);
        }

        #[allow(rustc::lint_pass_impl_without_macro)]
        impl LintPass for $name {
            fn name(&self) -> &'static str {
                panic!()
            }
        }
    )
}

/// A lint pass boxed up as a trait object.
pub type EarlyLintPassObject = Box<dyn EarlyLintPass + 'static>;
pub type LateLintPassObject<'tcx> = Box<dyn LateLintPass<'tcx> + 'tcx>;
