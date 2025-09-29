use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{is_bool, sym};
use rustc_abi::ExternAbi;
use rustc_hir as hir;
use rustc_hir::{FnSig, ImplItem};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::edition::Edition::{self, Edition2015, Edition2021};
use rustc_span::{Symbol, kw};

use super::SHOULD_IMPLEMENT_TRAIT;
use super::lib::SelfKind;

pub(super) fn check_impl_item<'tcx>(
    cx: &LateContext<'tcx>,
    impl_item: &'tcx ImplItem<'_>,
    self_ty: Ty<'tcx>,
    impl_implements_trait: bool,
    first_arg_ty_opt: Option<Ty<'tcx>>,
    sig: &FnSig<'_>,
) {
    // if this impl block implements a trait, lint in trait definition instead
    if !impl_implements_trait && cx.effective_visibilities.is_exported(impl_item.owner_id.def_id) {
        // check missing trait implementations
        for method_config in &TRAIT_METHODS {
            if impl_item.ident.name == method_config.method_name
                && sig.decl.inputs.len() == method_config.param_count
                && method_config.output_type.matches(&sig.decl.output)
                // in case there is no first arg, since we already have checked the number of arguments
                // it's should be always true
                && first_arg_ty_opt
                    .is_none_or(|first_arg_ty| method_config.self_kind.matches(cx, self_ty, first_arg_ty))
                && fn_header_equals(method_config.fn_header, sig.header)
                && method_config.lifetime_param_cond(impl_item)
                && method_config.in_prelude_since <= cx.tcx.sess.edition()
            {
                span_lint_and_help(
                    cx,
                    SHOULD_IMPLEMENT_TRAIT,
                    impl_item.span,
                    format!(
                        "method `{}` can be confused for the standard trait method `{}::{}`",
                        method_config.method_name, method_config.trait_name, method_config.method_name
                    ),
                    None,
                    format!(
                        "consider implementing the trait `{}` or choosing a less ambiguous method name",
                        method_config.trait_name
                    ),
                );
            }
        }
    }
}

const FN_HEADER: hir::FnHeader = hir::FnHeader {
    safety: hir::HeaderSafety::Normal(hir::Safety::Safe),
    constness: hir::Constness::NotConst,
    asyncness: hir::IsAsync::NotAsync,
    abi: ExternAbi::Rust,
};

struct ShouldImplTraitCase {
    trait_name: &'static str,
    method_name: Symbol,
    param_count: usize,
    fn_header: hir::FnHeader,
    // implicit self kind expected (none, self, &self, ...)
    self_kind: SelfKind,
    // checks against the output type
    output_type: OutType,
    // certain methods with explicit lifetimes can't implement the equivalent trait method
    lint_explicit_lifetime: bool,
    in_prelude_since: Edition,
}
impl ShouldImplTraitCase {
    #[expect(clippy::too_many_arguments)]
    const fn new(
        trait_name: &'static str,
        method_name: Symbol,
        param_count: usize,
        fn_header: hir::FnHeader,
        self_kind: SelfKind,
        output_type: OutType,
        lint_explicit_lifetime: bool,
        in_prelude_since: Edition,
    ) -> ShouldImplTraitCase {
        ShouldImplTraitCase {
            trait_name,
            method_name,
            param_count,
            fn_header,
            self_kind,
            output_type,
            lint_explicit_lifetime,
            in_prelude_since,
        }
    }

    fn lifetime_param_cond(&self, impl_item: &ImplItem<'_>) -> bool {
        self.lint_explicit_lifetime
            || !impl_item.generics.params.iter().any(|p| {
                matches!(
                    p.kind,
                    hir::GenericParamKind::Lifetime {
                        kind: hir::LifetimeParamKind::Explicit
                    }
                )
            })
    }
}

#[rustfmt::skip]
const TRAIT_METHODS: [ShouldImplTraitCase; 30] = [
    ShouldImplTraitCase::new("std::ops::Add",           sym::add,        2,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::convert::AsMut",     sym::as_mut,     1,  FN_HEADER, SelfKind::RefMut, OutType::Ref,  true,  Edition2015),
    ShouldImplTraitCase::new("std::convert::AsRef",     sym::as_ref,     1,  FN_HEADER, SelfKind::Ref,    OutType::Ref,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::BitAnd",        sym::bitand,     2,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::BitOr",         sym::bitor,      2,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::BitXor",        sym::bitxor,     2,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::borrow::Borrow",     sym::borrow,     1,  FN_HEADER, SelfKind::Ref,    OutType::Ref,  true,  Edition2015),
    ShouldImplTraitCase::new("std::borrow::BorrowMut",  sym::borrow_mut, 1,  FN_HEADER, SelfKind::RefMut, OutType::Ref,  true,  Edition2015),
    ShouldImplTraitCase::new("std::clone::Clone",       sym::clone,      1,  FN_HEADER, SelfKind::Ref,    OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::cmp::Ord",           sym::cmp,        2,  FN_HEADER, SelfKind::Ref,    OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::default::Default",   kw::Default,     0,  FN_HEADER, SelfKind::No,     OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::Deref",         sym::deref,      1,  FN_HEADER, SelfKind::Ref,    OutType::Ref,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::DerefMut",      sym::deref_mut,  1,  FN_HEADER, SelfKind::RefMut, OutType::Ref,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::Div",           sym::div,        2,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::Drop",          sym::drop,       1,  FN_HEADER, SelfKind::RefMut, OutType::Unit, true,  Edition2015),
    ShouldImplTraitCase::new("std::cmp::PartialEq",     sym::eq,         2,  FN_HEADER, SelfKind::Ref,    OutType::Bool, true,  Edition2015),
    ShouldImplTraitCase::new("std::iter::FromIterator", sym::from_iter,  1,  FN_HEADER, SelfKind::No,     OutType::Any,  true,  Edition2021),
    ShouldImplTraitCase::new("std::str::FromStr",       sym::from_str,   1,  FN_HEADER, SelfKind::No,     OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::hash::Hash",         sym::hash,       2,  FN_HEADER, SelfKind::Ref,    OutType::Unit, true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::Index",         sym::index,      2,  FN_HEADER, SelfKind::Ref,    OutType::Ref,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::IndexMut",      sym::index_mut,  2,  FN_HEADER, SelfKind::RefMut, OutType::Ref,  true,  Edition2015),
    ShouldImplTraitCase::new("std::iter::IntoIterator", sym::into_iter,  1,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::Mul",           sym::mul,        2,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::Neg",           sym::neg,        1,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::iter::Iterator",     sym::next,       1,  FN_HEADER, SelfKind::RefMut, OutType::Any,  false, Edition2015),
    ShouldImplTraitCase::new("std::ops::Not",           sym::not,        1,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::Rem",           sym::rem,        2,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::Shl",           sym::shl,        2,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::Shr",           sym::shr,        2,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
    ShouldImplTraitCase::new("std::ops::Sub",           sym::sub,        2,  FN_HEADER, SelfKind::Value,  OutType::Any,  true,  Edition2015),
];

#[derive(Clone, Copy)]
enum OutType {
    Unit,
    Bool,
    Any,
    Ref,
}

impl OutType {
    fn matches(self, ty: &hir::FnRetTy<'_>) -> bool {
        let is_unit = |ty: &hir::Ty<'_>| matches!(ty.kind, hir::TyKind::Tup(&[]));
        match (self, ty) {
            (Self::Unit, &hir::FnRetTy::DefaultReturn(_)) => true,
            (Self::Unit, &hir::FnRetTy::Return(ty)) if is_unit(ty) => true,
            (Self::Bool, &hir::FnRetTy::Return(ty)) if is_bool(ty) => true,
            (Self::Any, &hir::FnRetTy::Return(ty)) if !is_unit(ty) => true,
            (Self::Ref, &hir::FnRetTy::Return(ty)) => matches!(ty.kind, hir::TyKind::Ref(_, _)),
            _ => false,
        }
    }
}

fn fn_header_equals(expected: hir::FnHeader, actual: hir::FnHeader) -> bool {
    expected.constness == actual.constness && expected.safety == actual.safety && expected.asyncness == actual.asyncness
}
