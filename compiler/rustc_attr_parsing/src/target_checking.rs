use std::borrow::Cow;

use rustc_ast::AttrStyle;
use rustc_errors::DiagArgValue;
use rustc_feature::Features;
use rustc_hir::lints::AttributeLintKind;
use rustc_hir::{MethodKind, Target};

use crate::AttributeParser;
use crate::context::{AcceptContext, Stage};
use crate::session_diagnostics::InvalidTarget;

#[derive(Debug)]
pub(crate) enum AllowedTargets {
    AllowList(&'static [Policy]),
    AllowListWarnRest(&'static [Policy]),
    /// Special, and not the same as `AllowList(&[Allow(Target::Crate)])`.
    /// For crate-level attributes we emit a specific set of lints to warn
    /// people about accidentally not using them on the crate.
    /// Only use this for attributes that are *exclusively* valid at the crate level.
    CrateLevel,
}

pub(crate) enum AllowedResult {
    Allowed,
    Warn,
    Error,
}

impl AllowedTargets {
    pub(crate) fn is_allowed(&self, target: Target) -> AllowedResult {
        match self {
            AllowedTargets::AllowList(list) => {
                if list.contains(&Policy::Allow(target)) {
                    AllowedResult::Allowed
                } else if list.contains(&Policy::Warn(target)) {
                    AllowedResult::Warn
                } else {
                    AllowedResult::Error
                }
            }
            AllowedTargets::AllowListWarnRest(list) => {
                if list.contains(&Policy::Allow(target)) {
                    AllowedResult::Allowed
                } else if list.contains(&Policy::Error(target)) {
                    AllowedResult::Error
                } else {
                    AllowedResult::Warn
                }
            }
            AllowedTargets::CrateLevel => AllowedResult::Allowed,
        }
    }

    pub(crate) fn allowed_targets(&self) -> Vec<Target> {
        match self {
            AllowedTargets::AllowList(list) => list,
            AllowedTargets::AllowListWarnRest(list) => list,
            AllowedTargets::CrateLevel => ALL_TARGETS,
        }
        .iter()
        .filter_map(|target| match target {
            Policy::Allow(target) => Some(*target),
            Policy::Warn(_) => None,
            Policy::Error(_) => None,
        })
        .collect()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Policy {
    Allow(Target),
    Warn(Target),
    Error(Target),
}

impl<'sess, S: Stage> AttributeParser<'sess, S> {
    pub(crate) fn check_target(
        allowed_targets: &AllowedTargets,
        target: Target,
        cx: &mut AcceptContext<'_, 'sess, S>,
    ) {
        Self::check_type(matches!(allowed_targets, AllowedTargets::CrateLevel), target, cx);

        match allowed_targets.is_allowed(target) {
            AllowedResult::Allowed => {}
            AllowedResult::Warn => {
                let allowed_targets = allowed_targets.allowed_targets();
                let (applied, only) = allowed_targets_applied(allowed_targets, target, cx.features);
                let name = cx.attr_path.clone();
                let attr_span = cx.attr_span;
                cx.emit_lint(
                    AttributeLintKind::InvalidTarget {
                        name,
                        target,
                        only: if only { "only " } else { "" },
                        applied,
                    },
                    attr_span,
                );
            }
            AllowedResult::Error => {
                let allowed_targets = allowed_targets.allowed_targets();
                let (applied, only) = allowed_targets_applied(allowed_targets, target, cx.features);
                let name = cx.attr_path.clone();
                cx.dcx().emit_err(InvalidTarget {
                    span: cx.attr_span.clone(),
                    name,
                    target: target.plural_name(),
                    only: if only { "only " } else { "" },
                    applied: DiagArgValue::StrListSepByAnd(
                        applied.into_iter().map(Cow::Owned).collect(),
                    ),
                });
            }
        }
    }

    pub(crate) fn check_type(
        crate_level: bool,
        target: Target,
        cx: &mut AcceptContext<'_, 'sess, S>,
    ) {
        let is_crate_root = S::id_is_crate_root(cx.target_id);

        if is_crate_root {
            return;
        }

        if !crate_level {
            return;
        }

        let lint = AttributeLintKind::InvalidStyle {
            name: cx.attr_path.clone(),
            is_used_as_inner: cx.attr_style == AttrStyle::Inner,
            target,
            target_span: cx.target_span,
        };
        let attr_span = cx.attr_span;

        cx.emit_lint(lint, attr_span);
    }
}

/// Takes a list of `allowed_targets` for an attribute, and the `target` the attribute was applied to.
/// Does some heuristic-based filtering to remove uninteresting targets, and formats the targets into a string
pub(crate) fn allowed_targets_applied(
    mut allowed_targets: Vec<Target>,
    target: Target,
    features: Option<&Features>,
) -> (Vec<String>, bool) {
    // Remove unstable targets from `allowed_targets` if their features are not enabled
    if let Some(features) = features {
        if !features.fn_delegation() {
            allowed_targets.retain(|t| !matches!(t, Target::Delegation { .. }));
        }
        if !features.stmt_expr_attributes() {
            allowed_targets.retain(|t| !matches!(t, Target::Expression | Target::Statement));
        }
        if !features.extern_types() {
            allowed_targets.retain(|t| !matches!(t, Target::ForeignTy));
        }
    }

    // We define groups of "similar" targets.
    // If at least two of the targets are allowed, and the `target` is not in the group,
    // we collapse the entire group to a single entry to simplify the target list
    const FUNCTION_LIKE: &[Target] = &[
        Target::Fn,
        Target::Closure,
        Target::ForeignFn,
        Target::Method(MethodKind::Inherent),
        Target::Method(MethodKind::Trait { body: false }),
        Target::Method(MethodKind::Trait { body: true }),
        Target::Method(MethodKind::TraitImpl),
    ];
    const METHOD_LIKE: &[Target] = &[
        Target::Method(MethodKind::Inherent),
        Target::Method(MethodKind::Trait { body: false }),
        Target::Method(MethodKind::Trait { body: true }),
        Target::Method(MethodKind::TraitImpl),
    ];
    const IMPL_LIKE: &[Target] =
        &[Target::Impl { of_trait: false }, Target::Impl { of_trait: true }];
    const ADT_LIKE: &[Target] = &[Target::Struct, Target::Enum];

    let mut added_fake_targets = Vec::new();
    filter_targets(
        &mut allowed_targets,
        FUNCTION_LIKE,
        "functions",
        target,
        &mut added_fake_targets,
    );
    filter_targets(&mut allowed_targets, METHOD_LIKE, "methods", target, &mut added_fake_targets);
    filter_targets(&mut allowed_targets, IMPL_LIKE, "impl blocks", target, &mut added_fake_targets);
    filter_targets(&mut allowed_targets, ADT_LIKE, "data types", target, &mut added_fake_targets);

    // If there is now only 1 target left, show that as the only possible target
    (
        added_fake_targets
            .iter()
            .copied()
            .chain(allowed_targets.iter().map(|t| t.plural_name()))
            .map(|i| i.to_string())
            .collect(),
        allowed_targets.len() + added_fake_targets.len() == 1,
    )
}

fn filter_targets(
    allowed_targets: &mut Vec<Target>,
    target_group: &'static [Target],
    target_group_name: &'static str,
    target: Target,
    added_fake_targets: &mut Vec<&'static str>,
) {
    if target_group.contains(&target) {
        return;
    }
    if allowed_targets.iter().filter(|at| target_group.contains(at)).count() < 2 {
        return;
    }
    allowed_targets.retain(|t| !target_group.contains(t));
    added_fake_targets.push(target_group_name);
}

/// This is the list of all targets to which a attribute can be applied
/// This is used for:
/// - `rustc_dummy`, which can be applied to all targets
/// - Attributes that are not parted to the new target system yet can use this list as a placeholder
pub(crate) const ALL_TARGETS: &'static [Policy] = {
    use Policy::Allow;
    &[
        Allow(Target::ExternCrate),
        Allow(Target::Use),
        Allow(Target::Static),
        Allow(Target::Const),
        Allow(Target::Fn),
        Allow(Target::Closure),
        Allow(Target::Mod),
        Allow(Target::ForeignMod),
        Allow(Target::GlobalAsm),
        Allow(Target::TyAlias),
        Allow(Target::Enum),
        Allow(Target::Variant),
        Allow(Target::Struct),
        Allow(Target::Field),
        Allow(Target::Union),
        Allow(Target::Trait),
        Allow(Target::TraitAlias),
        Allow(Target::Impl { of_trait: false }),
        Allow(Target::Impl { of_trait: true }),
        Allow(Target::Expression),
        Allow(Target::Statement),
        Allow(Target::Arm),
        Allow(Target::AssocConst),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::AssocTy),
        Allow(Target::ForeignFn),
        Allow(Target::ForeignStatic),
        Allow(Target::ForeignTy),
        Allow(Target::MacroDef),
        Allow(Target::Param),
        Allow(Target::PatField),
        Allow(Target::ExprField),
        Allow(Target::WherePredicate),
        Allow(Target::MacroCall),
        Allow(Target::Crate),
        Allow(Target::Delegation { mac: false }),
        Allow(Target::Delegation { mac: true }),
    ]
};
