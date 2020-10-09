use crate::context::{EarlyContext, LateContext};

use rustc_ast as ast;
use rustc_data_structures::sync;
use rustc_hir as hir;
use rustc_session::lint::builtin::HardwiredLints;
use rustc_session::lint::LintPass;
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::Span;

#[macro_export]
macro_rules! late_lint_methods {
    ($macro:path, $args:tt, [$hir:tt]) => (
        $macro!($args, [$hir], [
            fn check_param(a: &$hir hir::Param<$hir>);
            fn check_body(a: &$hir hir::Body<$hir>);
            fn check_body_post(a: &$hir hir::Body<$hir>);
            fn check_name(a: Span, b: Symbol);
            fn check_crate(a: &$hir hir::Crate<$hir>);
            fn check_crate_post(a: &$hir hir::Crate<$hir>);
            fn check_mod(a: &$hir hir::Mod<$hir>, b: Span, c: hir::HirId);
            fn check_mod_post(a: &$hir hir::Mod<$hir>, b: Span, c: hir::HirId);
            fn check_foreign_item(a: &$hir hir::ForeignItem<$hir>);
            fn check_foreign_item_post(a: &$hir hir::ForeignItem<$hir>);
            fn check_item(a: &$hir hir::Item<$hir>);
            fn check_item_post(a: &$hir hir::Item<$hir>);
            fn check_local(a: &$hir hir::Local<$hir>);
            fn check_block(a: &$hir hir::Block<$hir>);
            fn check_block_post(a: &$hir hir::Block<$hir>);
            fn check_stmt(a: &$hir hir::Stmt<$hir>);
            fn check_arm(a: &$hir hir::Arm<$hir>);
            fn check_pat(a: &$hir hir::Pat<$hir>);
            fn check_expr(a: &$hir hir::Expr<$hir>);
            fn check_expr_post(a: &$hir hir::Expr<$hir>);
            fn check_ty(a: &$hir hir::Ty<$hir>);
            fn check_generic_arg(a: &$hir hir::GenericArg<$hir>);
            fn check_generic_param(a: &$hir hir::GenericParam<$hir>);
            fn check_generics(a: &$hir hir::Generics<$hir>);
            fn check_where_predicate(a: &$hir hir::WherePredicate<$hir>);
            fn check_poly_trait_ref(a: &$hir hir::PolyTraitRef<$hir>, b: hir::TraitBoundModifier);
            fn check_fn(
                a: rustc_hir::intravisit::FnKind<$hir>,
                b: &$hir hir::FnDecl<$hir>,
                c: &$hir hir::Body<$hir>,
                d: Span,
                e: hir::HirId);
            fn check_fn_post(
                a: rustc_hir::intravisit::FnKind<$hir>,
                b: &$hir hir::FnDecl<$hir>,
                c: &$hir hir::Body<$hir>,
                d: Span,
                e: hir::HirId
            );
            fn check_trait_item(a: &$hir hir::TraitItem<$hir>);
            fn check_trait_item_post(a: &$hir hir::TraitItem<$hir>);
            fn check_impl_item(a: &$hir hir::ImplItem<$hir>);
            fn check_impl_item_post(a: &$hir hir::ImplItem<$hir>);
            fn check_struct_def(a: &$hir hir::VariantData<$hir>);
            fn check_struct_def_post(a: &$hir hir::VariantData<$hir>);
            fn check_struct_field(a: &$hir hir::StructField<$hir>);
            fn check_variant(a: &$hir hir::Variant<$hir>);
            fn check_variant_post(a: &$hir hir::Variant<$hir>);
            fn check_lifetime(a: &$hir hir::Lifetime);
            fn check_path(a: &$hir hir::Path<$hir>, b: hir::HirId);
            fn check_attribute(a: &$hir ast::Attribute);

            /// Called when entering a syntax node that can have lint attributes such
            /// as `#[allow(...)]`. Called with *all* the attributes of that node.
            fn enter_lint_attrs(a: &$hir [ast::Attribute]);

            /// Counterpart to `enter_lint_attrs`.
            fn exit_lint_attrs(a: &$hir [ast::Attribute]);
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

macro_rules! expand_lint_pass_methods {
    ($context:ty, [$($(#[$attr:meta])* fn $name:ident($($param:ident: $arg:ty),*);)*]) => (
        $(#[inline(always)] fn $name(&mut self, _: $context, $(_: $arg),*) {})*
    )
}

macro_rules! declare_late_lint_pass {
    ([], [$hir:tt], [$($methods:tt)*]) => (
        pub trait LateLintPass<$hir>: LintPass {
            expand_lint_pass_methods!(&LateContext<$hir>, [$($methods)*]);
        }
    )
}

late_lint_methods!(declare_late_lint_pass, [], ['tcx]);

impl LateLintPass<'_> for HardwiredLints {}

#[macro_export]
macro_rules! expand_combined_late_lint_pass_method {
    ([$($passes:ident),*], $self: ident, $name: ident, $params:tt) => ({
        $($self.$passes.$name $params;)*
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

#[macro_export]
macro_rules! declare_combined_late_lint_pass {
    ([$v:vis $name:ident, [$($passes:ident: $constructor:expr,)*]], [$hir:tt], $methods:tt) => (
        #[allow(non_snake_case)]
        $v struct $name {
            $($passes: $passes,)*
        }

        impl $name {
            $v fn new() -> Self {
                Self {
                    $($passes: $constructor,)*
                }
            }

            $v fn get_lints() -> LintArray {
                let mut lints = Vec::new();
                $(lints.extend_from_slice(&$passes::get_lints());)*
                lints
            }
        }

        impl<'tcx> LateLintPass<'tcx> for $name {
            expand_combined_late_lint_pass_methods!([$($passes),*], $methods);
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
            fn check_mod(a: &ast::Mod, b: Span, c: ast::NodeId);
            fn check_mod_post(a: &ast::Mod, b: Span, c: ast::NodeId);
            fn check_foreign_item(a: &ast::ForeignItem);
            fn check_foreign_item_post(a: &ast::ForeignItem);
            fn check_item(a: &ast::Item);
            fn check_item_post(a: &ast::Item);
            fn check_local(a: &ast::Local);
            fn check_block(a: &ast::Block);
            fn check_block_post(a: &ast::Block);
            fn check_stmt(a: &ast::Stmt);
            fn check_arm(a: &ast::Arm);
            fn check_pat(a: &ast::Pat);
            fn check_anon_const(a: &ast::AnonConst);
            fn check_pat_post(a: &ast::Pat);
            fn check_expr(a: &ast::Expr);
            fn check_expr_post(a: &ast::Expr);
            fn check_ty(a: &ast::Ty);
            fn check_generic_arg(a: &ast::GenericArg);
            fn check_generic_param(a: &ast::GenericParam);
            fn check_generics(a: &ast::Generics);
            fn check_where_predicate(a: &ast::WherePredicate);
            fn check_poly_trait_ref(a: &ast::PolyTraitRef,
                                    b: &ast::TraitBoundModifier);
            fn check_fn(a: rustc_ast::visit::FnKind<'_>, c: Span, d_: ast::NodeId);
            fn check_fn_post(
                a: rustc_ast::visit::FnKind<'_>,
                c: Span,
                d: ast::NodeId
            );
            fn check_trait_item(a: &ast::AssocItem);
            fn check_trait_item_post(a: &ast::AssocItem);
            fn check_impl_item(a: &ast::AssocItem);
            fn check_impl_item_post(a: &ast::AssocItem);
            fn check_struct_def(a: &ast::VariantData);
            fn check_struct_def_post(a: &ast::VariantData);
            fn check_struct_field(a: &ast::StructField);
            fn check_variant(a: &ast::Variant);
            fn check_variant_post(a: &ast::Variant);
            fn check_lifetime(a: &ast::Lifetime);
            fn check_path(a: &ast::Path, b: ast::NodeId);
            fn check_attribute(a: &ast::Attribute);
            fn check_mac_def(a: &ast::MacroDef, b: ast::NodeId);
            fn check_mac(a: &ast::MacCall);

            /// Called when entering a syntax node that can have lint attributes such
            /// as `#[allow(...)]`. Called with *all* the attributes of that node.
            fn enter_lint_attrs(a: &[ast::Attribute]);

            /// Counterpart to `enter_lint_attrs`.
            fn exit_lint_attrs(a: &[ast::Attribute]);
        ]);
    )
}

macro_rules! expand_early_lint_pass_methods {
    ($context:ty, [$($(#[$attr:meta])* fn $name:ident($($param:ident: $arg:ty),*);)*]) => (
        $(#[inline(always)] fn $name(&mut self, _: $context, $(_: $arg),*) {})*
    )
}

macro_rules! declare_early_lint_pass {
    ([], [$($methods:tt)*]) => (
        pub trait EarlyLintPass: LintPass {
            expand_early_lint_pass_methods!(&EarlyContext<'_>, [$($methods)*]);
        }
    )
}

early_lint_methods!(declare_early_lint_pass, []);

#[macro_export]
macro_rules! expand_combined_early_lint_pass_method {
    ([$($passes:ident),*], $self: ident, $name: ident, $params:tt) => ({
        $($self.$passes.$name $params;)*
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

#[macro_export]
macro_rules! declare_combined_early_lint_pass {
    ([$v:vis $name:ident, [$($passes:ident: $constructor:expr,)*]], $methods:tt) => (
        #[allow(non_snake_case)]
        $v struct $name {
            $($passes: $passes,)*
        }

        impl $name {
            $v fn new() -> Self {
                Self {
                    $($passes: $constructor,)*
                }
            }

            $v fn get_lints() -> LintArray {
                let mut lints = Vec::new();
                $(lints.extend_from_slice(&$passes::get_lints());)*
                lints
            }
        }

        impl EarlyLintPass for $name {
            expand_combined_early_lint_pass_methods!([$($passes),*], $methods);
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
pub type EarlyLintPassObject = Box<dyn EarlyLintPass + sync::Send + sync::Sync + 'static>;
pub type LateLintPassObject =
    Box<dyn for<'tcx> LateLintPass<'tcx> + sync::Send + sync::Sync + 'static>;
