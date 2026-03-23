use std::borrow::Cow;

use rustc_ast::AttrStyle;
use rustc_errors::{DiagArgValue, MultiSpan, StashKey};
use rustc_feature::Features;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::lints::AttributeLintKind;
use rustc_hir::{AttrItem, Attribute, MethodKind, Target};
use rustc_span::{BytePos, Span, Symbol, sym};

use crate::AttributeParser;
use crate::context::{AcceptContext, Stage};
use crate::errors::{
    InvalidAttrAtCrateLevel, ItemFollowingInnerAttr, UnsupportedAttributesInWhere,
};
use crate::session_diagnostics::InvalidTarget;
use crate::target_checking::Policy::Allow;

#[derive(Debug)]
pub(crate) enum AllowedTargets {
    AllowList(&'static [Policy]),
    AllowListWarnRest(&'static [Policy]),
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
                if list.contains(&Policy::Allow(target))
                    || list.contains(&Policy::AllowSilent(target))
                {
                    AllowedResult::Allowed
                } else if list.contains(&Policy::Warn(target)) {
                    AllowedResult::Warn
                } else {
                    AllowedResult::Error
                }
            }
            AllowedTargets::AllowListWarnRest(list) => {
                if list.contains(&Policy::Allow(target))
                    || list.contains(&Policy::AllowSilent(target))
                {
                    AllowedResult::Allowed
                } else if list.contains(&Policy::Error(target)) {
                    AllowedResult::Error
                } else {
                    AllowedResult::Warn
                }
            }
        }
    }

    pub(crate) fn allowed_targets(&self) -> Vec<Target> {
        match self {
            AllowedTargets::AllowList(list) => list,
            AllowedTargets::AllowListWarnRest(list) => list,
        }
        .iter()
        .filter_map(|target| match target {
            Policy::Allow(target) => Some(*target),
            Policy::AllowSilent(_) => None, // Not listed in possible targets
            Policy::Warn(_) => None,
            Policy::Error(_) => None,
        })
        .collect()
    }
}

/// This policy determines what diagnostics should be emitted based on the `Target` of the attribute.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Policy {
    /// A target that is allowed.
    Allow(Target),
    /// A target that is allowed and not listed in the possible targets.
    /// This is useful if the target is checked elsewhere.
    AllowSilent(Target),
    /// Emits a FCW on this target.
    /// This is useful if the target was previously allowed but should not be.
    Warn(Target),
    /// Emits an error on this target.
    Error(Target),
}

impl<'sess, S: Stage> AttributeParser<'sess, S> {
    pub(crate) fn check_target(
        allowed_targets: &AllowedTargets,
        target: Target,
        cx: &mut AcceptContext<'_, 'sess, S>,
    ) {
        // For crate-level attributes we emit a specific set of lints to warn
        // people about accidentally not using them on the crate.
        if let &AllowedTargets::AllowList(&[Allow(Target::Crate)]) = allowed_targets {
            Self::check_crate_level(target, cx);
            return;
        }

        if matches!(cx.attr_path.segments.as_ref(), [sym::repr]) && target == Target::Crate {
            // The allowed targets of `repr` depend on its arguments. They can't be checked using
            // the `AttributeParser` code.
            let span = cx.attr_span;
            let item =
                cx.cx.first_line_of_next_item(span).map(|span| ItemFollowingInnerAttr { span });

            let pound_to_opening_bracket = cx.attr_span.until(cx.inner_span);

            cx.dcx()
                .create_err(InvalidAttrAtCrateLevel {
                    span,
                    pound_to_opening_bracket,
                    name: sym::repr,
                    item,
                })
                .emit();
        }

        match allowed_targets.is_allowed(target) {
            AllowedResult::Allowed => {}
            AllowedResult::Warn => {
                let allowed_targets = allowed_targets.allowed_targets();
                let (applied, only) = allowed_targets_applied(allowed_targets, target, cx.features);
                let name = cx.attr_path.clone();

                let lint = if name.segments[0] == sym::deprecated
                    && ![
                        Target::Closure,
                        Target::Expression,
                        Target::Statement,
                        Target::Arm,
                        Target::MacroCall,
                    ]
                    .contains(&target)
                {
                    rustc_session::lint::builtin::USELESS_DEPRECATED
                } else {
                    rustc_session::lint::builtin::UNUSED_ATTRIBUTES
                };

                let attr_span = cx.attr_span;
                cx.emit_lint(
                    lint,
                    AttributeLintKind::InvalidTarget {
                        name: name.to_string(),
                        target: target.plural_name(),
                        only: if only { "only " } else { "" },
                        applied,
                        attr_span,
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

    pub(crate) fn check_crate_level(target: Target, cx: &mut AcceptContext<'_, 'sess, S>) {
        if target == Target::Crate {
            return;
        }

        let kind = AttributeLintKind::InvalidStyle {
            name: cx.attr_path.to_string(),
            is_used_as_inner: cx.attr_style == AttrStyle::Inner,
            target: target.name(),
            target_span: cx.target_span,
        };
        let attr_span = cx.attr_span;

        cx.emit_lint(rustc_session::lint::builtin::UNUSED_ATTRIBUTES, kind, attr_span);
    }

    // FIXME: Fix "Cannot determine resolution" error and remove built-in macros
    // from this check.
    pub(crate) fn check_invalid_crate_level_attr_item(&self, attr: &AttrItem, inner_span: Span) {
        // Check for builtin attributes at the crate level
        // which were unsuccessfully resolved due to cannot determine
        // resolution for the attribute macro error.
        const ATTRS_TO_CHECK: &[Symbol] =
            &[sym::derive, sym::test, sym::test_case, sym::global_allocator, sym::bench];

        // FIXME(jdonszelmann): all attrs should be combined here cleaning this up some day.
        if let Some(name) = ATTRS_TO_CHECK.iter().find(|attr_to_check| matches!(attr.path.segments.as_ref(), [segment] if segment == *attr_to_check)) {
            let span = attr.span;
            let name = *name;

            let item = self.first_line_of_next_item(span).map(|span| ItemFollowingInnerAttr { span });

            let err = self.dcx().create_err(InvalidAttrAtCrateLevel {
                span,
                pound_to_opening_bracket: span.until(inner_span),
                name,
                item,
            });

            self.dcx().try_steal_replace_and_emit_err(
                attr.path.span,
                StashKey::UndeterminedMacroResolution,
                err,
            );
        }
    }

    fn first_line_of_next_item(&self, span: Span) -> Option<Span> {
        // We can't exactly call `tcx.hir_free_items()` here because it's too early and querying
        // this would create a circular dependency. Instead, we resort to getting the original
        // source code that follows `span` and find the next item from here.

        self.sess()
            .source_map()
            .span_to_source(span, |content, _, span_end| {
                let mut source = &content[span_end..];
                let initial_source_len = source.len();
                let span = try {
                    loop {
                        let first = source.chars().next()?;

                        if first.is_whitespace() {
                            let split_idx = source.find(|c: char| !c.is_whitespace())?;
                            source = &source[split_idx..];
                        } else if source.starts_with("//") {
                            let line_idx = source.find('\n')?;
                            source = &source[line_idx + '\n'.len_utf8()..];
                        } else if source.starts_with("/*") {
                            // FIXME: support nested comments.
                            let close_idx = source.find("*/")?;
                            source = &source[close_idx + "*/".len()..];
                        } else if first == '#' {
                            // FIXME: properly find the end of the attributes in order to accurately
                            // skip them. This version just consumes the source code until the next
                            // `]`.
                            let close_idx = source.find(']')?;
                            source = &source[close_idx + ']'.len_utf8()..];
                        } else {
                            let lo = span_end + initial_source_len - source.len();
                            let last_line = source.split('\n').next().map(|s| s.trim_end())?;

                            let hi = lo + last_line.len();
                            let lo = BytePos(lo as u32);
                            let hi = BytePos(hi as u32);
                            let next_item_span = Span::new(lo, hi, span.ctxt(), None);

                            break next_item_span;
                        }
                    }
                };

                Ok(span)
            })
            .ok()
            .flatten()
    }

    pub(crate) fn check_invalid_where_predicate_attrs<'attr>(
        &self,
        attrs: impl IntoIterator<Item = &'attr Attribute>,
    ) {
        // FIXME(where_clause_attrs): Currently, as the following check shows,
        // only `#[cfg]` and `#[cfg_attr]` are allowed, but it should be removed
        // if we allow more attributes (e.g., tool attributes and `allow/deny/warn`)
        // in where clauses. After that, this function would become useless.
        let spans = attrs
            .into_iter()
            // FIXME: We shouldn't need to special-case `doc`!
            .filter(|attr| {
                matches!(
                    attr,
                    Attribute::Parsed(AttributeKind::DocComment { .. } | AttributeKind::Doc(_))
                        | Attribute::Unparsed(_)
                )
            })
            .map(|attr| attr.span())
            .collect::<Vec<_>>();
        if !spans.is_empty() {
            self.dcx()
                .emit_err(UnsupportedAttributesInWhere { span: MultiSpan::from_spans(spans) });
        }
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
    const ADT_LIKE: &[Target] = &[Target::Struct, Target::Enum, Target::Union];

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

    let mut target_strings: Vec<_> = added_fake_targets
        .iter()
        .copied()
        .chain(allowed_targets.iter().map(|t| t.plural_name()))
        .map(|i| i.to_string())
        .collect();

    // ensure a consistent order
    target_strings.sort();

    // If there is now only 1 target left, show that as the only possible target
    let only_target = target_strings.len() == 1;

    (target_strings, only_target)
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
        Allow(Target::GenericParam {
            kind: rustc_hir::target::GenericParamKind::Const,
            has_default: false,
        }),
        Allow(Target::GenericParam {
            kind: rustc_hir::target::GenericParamKind::Const,
            has_default: true,
        }),
        Allow(Target::GenericParam {
            kind: rustc_hir::target::GenericParamKind::Lifetime,
            has_default: false,
        }),
        Allow(Target::GenericParam {
            kind: rustc_hir::target::GenericParamKind::Lifetime,
            has_default: true,
        }),
        Allow(Target::GenericParam {
            kind: rustc_hir::target::GenericParamKind::Type,
            has_default: false,
        }),
        Allow(Target::GenericParam {
            kind: rustc_hir::target::GenericParamKind::Type,
            has_default: true,
        }),
    ]
};
