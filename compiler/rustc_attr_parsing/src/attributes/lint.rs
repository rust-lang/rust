use rustc_ast::LitKind;
use rustc_hir::HashIgnoredAttrId;
use rustc_hir::attrs::{LintAttribute, LintAttributeKind, LintInstance};
use rustc_hir::lints::AttributeLintKind;
use rustc_hir::target::GenericParamKind;
use rustc_session::DynLintStore;
use rustc_session::lint::builtin::{RENAMED_AND_REMOVED_LINTS, UNKNOWN_LINTS, UNUSED_ATTRIBUTES};
use rustc_session::lint::{CheckLintNameResult, LintId};

use super::prelude::*;
use crate::attributes::AcceptFn;
use crate::session_diagnostics::UnknownToolInScopedLint;

pub(crate) trait Lint {
    const KIND: LintAttributeKind;
    const ATTR_SYMBOL: Symbol = Self::KIND.symbol();
}

pub(crate) struct Allow;

impl Lint for Allow {
    const KIND: LintAttributeKind = LintAttributeKind::Allow;
}
pub(crate) struct Deny;

impl Lint for Deny {
    const KIND: LintAttributeKind = LintAttributeKind::Deny;
}
pub(crate) struct Expect;

impl Lint for Expect {
    const KIND: LintAttributeKind = LintAttributeKind::Expect;
}
pub(crate) struct Forbid;

impl Lint for Forbid {
    const KIND: LintAttributeKind = LintAttributeKind::Forbid;
}
pub(crate) struct Warn;

impl Lint for Warn {
    const KIND: LintAttributeKind = LintAttributeKind::Warn;
}

#[derive(Default)]
pub(crate) struct LintParser {
    lint_attrs: ThinVec<LintAttribute>,
}

trait Mapping<S: Stage> {
    const MAPPING: (&'static [Symbol], AttributeTemplate, AcceptFn<LintParser, S>);
}
impl<S: Stage, T: Lint> Mapping<S> for T {
    const MAPPING: (&'static [Symbol], AttributeTemplate, AcceptFn<LintParser, S>) = (
        &[T::ATTR_SYMBOL],
        template!(
            List: &["lint1", "lint1, lint2, ...", r#"lint1, lint2, lint3, reason = "...""#],
            "https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes"
        ),
        |this, cx, args| {
            if let Some(lint_attr) = validate_lint_attr::<T, S>(cx, args) {
                this.lint_attrs.push(lint_attr);
            }
        },
    );
}

impl<S: Stage> AttributeParser<S> for LintParser {
    const ATTRIBUTES: AcceptMapping<Self, S> =
        &[Allow::MAPPING, Deny::MAPPING, Expect::MAPPING, Forbid::MAPPING, Warn::MAPPING];

    const ALLOWED_TARGETS: AllowedTargets = {
        use super::prelude::{Allow, Warn};
        AllowedTargets::AllowList(&[
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
            Allow(Target::Crate),
            Allow(Target::Delegation { mac: false }),
            Allow(Target::Delegation { mac: true }),
            Allow(Target::GenericParam { kind: GenericParamKind::Type, has_default: false }),
            Allow(Target::GenericParam { kind: GenericParamKind::Lifetime, has_default: false }),
            Allow(Target::GenericParam { kind: GenericParamKind::Const, has_default: false }),
            Allow(Target::GenericParam { kind: GenericParamKind::Type, has_default: true }),
            Allow(Target::GenericParam { kind: GenericParamKind::Lifetime, has_default: true }),
            Allow(Target::GenericParam { kind: GenericParamKind::Const, has_default: true }),
            Warn(Target::MacroCall),
        ])
    };

    fn finalize(mut self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if !self.lint_attrs.is_empty() {
            // Sort to ensure correct order operations later
            self.lint_attrs.sort_by(|a, b| a.attr_span.cmp(&b.attr_span));
            Some(AttributeKind::LintAttributes(self.lint_attrs))
        } else {
            None
        }
    }
}

#[inline(always)]
fn validate_lint_attr<T: Lint, S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    args: &ArgParser,
) -> Option<LintAttribute> {
    let Some(lint_store) = cx.sess.lint_store.as_ref().map(|store| store.to_owned()) else {
        unreachable!("lint_store required while parsing attributes");
    };
    let lint_store = lint_store.as_ref();
    let Some(list) = args.list() else {
        let span = cx.inner_span;
        cx.adcx().expected_list(span, args);
        return None;
    };
    let mut list = list.mixed().peekable();

    let mut skip_unused_check = false;
    let mut errored = false;
    let mut reason = None;
    let mut lint_instances = ThinVec::new();
    let mut lint_index = 0;
    let targeting_crate = matches!(cx.target, Target::Crate);
    while let Some(item) = list.next() {
        let Some(meta_item) = item.meta_item() else {
            cx.adcx().expected_identifier(item.span());
            errored = true;
            continue;
        };

        match meta_item.args() {
            ArgParser::NameValue(nv_parser) if meta_item.path().word_is(sym::reason) => {
                //FIXME replace this with duplicate check?
                if list.peek().is_some() {
                    cx.adcx().expected_nv_as_last_argument(meta_item.span(), sym::reason);
                    errored = true;
                    continue;
                }

                let val_lit = nv_parser.value_as_lit();
                let LitKind::Str(reason_sym, _) = val_lit.kind else {
                    cx.adcx().expected_string_literal(nv_parser.value_span, Some(val_lit));
                    errored = true;
                    continue;
                };
                reason = Some(reason_sym);
            }
            ArgParser::NameValue(_) => {
                cx.adcx().expected_specific_argument(meta_item.span(), &[sym::reason]);
                errored = true;
            }
            ArgParser::List(list) => {
                cx.adcx().expected_no_args(list.span);
                errored = true;
            }
            ArgParser::NoArgs => {
                skip_unused_check = true;
                let mut segments = meta_item.path().segments();

                let Some(tool_or_name) = segments.next() else {
                    unreachable!("first segment should always exist");
                };

                let rest = segments.collect::<Vec<_>>();
                let (tool_name, tool_span, name): (Option<Symbol>, Option<Span>, _) =
                    if rest.is_empty() {
                        let name = tool_or_name.name;
                        (None, None, name.to_string())
                    } else {
                        let tool = tool_or_name;
                        let name = rest
                            .into_iter()
                            .map(|ident| ident.to_string())
                            .collect::<Vec<_>>()
                            .join("::");
                        (Some(tool.name), Some(tool.span), name)
                    };

                let meta_item_span = meta_item.span();
                let original_name = Symbol::intern(&name);
                let mut full_name = tool_name
                    .map(|tool| Symbol::intern(&format!("{tool}::{}", original_name)))
                    .unwrap_or(original_name);

                if let Some(ids) = check_lint(
                    cx,
                    lint_store,
                    original_name,
                    &mut full_name,
                    tool_name,
                    tool_span,
                    meta_item_span,
                ) {
                    if !targeting_crate && ids.iter().any(|lint_id| lint_id.lint.crate_level_only) {
                        cx.emit_lint(
                            UNUSED_ATTRIBUTES,
                            AttributeLintKind::IgnoredUnlessCrateSpecified {
                                level: T::ATTR_SYMBOL,
                                name: original_name,
                            },
                            meta_item_span,
                        );
                    }
                    lint_instances.extend(ids.into_iter().map(|id| {
                        LintInstance::new(
                            full_name,
                            Symbol::intern(&id.to_string()),
                            meta_item_span,
                            lint_index,
                        )
                    }));
                }
                lint_index += 1;
            }
        }
    }

    if errored {
        return None;
    }

    if !skip_unused_check && lint_instances.is_empty() {
        let span = cx.attr_span;
        cx.adcx().warn_empty_attribute(span);
    }

    Some(LintAttribute {
        reason,
        lint_instances,
        attr_span: cx.attr_span,
        attr_style: cx.attr_style,
        attr_id: HashIgnoredAttrId { attr_id: cx.attr_id },
        kind: T::KIND,
    })
}

fn check_lint<'a, S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    lint_store: &'a dyn DynLintStore,
    original_name: Symbol,
    full_name: &mut Symbol,
    tool_name: Option<Symbol>,
    tool_span: Option<Span>,
    span: Span,
) -> Option<&'a [LintId]> {
    let Some(tools) = cx.tools else {
        unreachable!("tools required while parsing attributes");
    };
    if tools.is_empty() {
        unreachable!("tools should never be empty")
    }

    match lint_store.check_lint_name(original_name.as_str(), tool_name, tools) {
        CheckLintNameResult::Ok(ids) => Some(ids),
        CheckLintNameResult::Tool(ids, new_lint_name) => {
            let _name = match new_lint_name {
                None => original_name,
                Some(new_lint_name) => {
                    let new_lint_name = Symbol::intern(&new_lint_name);
                    cx.emit_lint(
                        RENAMED_AND_REMOVED_LINTS,
                        AttributeLintKind::DeprecatedLintName {
                            name: *full_name,
                            suggestion: span,
                            replace: new_lint_name,
                        },
                        span,
                    );
                    new_lint_name
                }
            };
            Some(ids)
        }

        CheckLintNameResult::MissingTool => {
            // If `MissingTool` is returned, then either the lint does not
            // exist in the tool or the code was not compiled with the tool and
            // therefore the lint was never added to the `LintStore`. To detect
            // this is the responsibility of the lint tool.
            None
        }

        CheckLintNameResult::NoTool => {
            cx.emit_err(UnknownToolInScopedLint {
                span: tool_span,
                tool_name: tool_name.unwrap(),
                full_lint_name: *full_name,
                is_nightly_build: cx.sess.is_nightly_build(),
            });
            None
        }

        CheckLintNameResult::Renamed(replace) => {
            cx.emit_lint(
                RENAMED_AND_REMOVED_LINTS,
                AttributeLintKind::RenamedLint { name: *full_name, replace, suggestion: span },
                span,
            );

            // Since it was renamed, and we have emitted the warning
            // we replace the "full_name", to ensure we don't get notes with:
            // `#[allow(NEW_NAME)]` implied by `#[allow(OLD_NAME)]`
            // Other lints still have access to the original name as the user wrote it,
            // through `original_name`
            *full_name = replace;

            // If this lint was renamed, apply the new lint instead of ignoring the
            // attribute. Ignore any errors or warnings that happen because the new
            // name is inaccurate.
            // NOTE: `new_name` already includes the tool name, so we don't
            // have to add it again.
            match lint_store.check_lint_name(replace.as_str(), None, tools) {
                CheckLintNameResult::Ok(ids) => Some(ids),
                _ => panic!("renamed lint does not exist: {replace}"),
            }
        }

        CheckLintNameResult::RenamedToolLint(new_name) => {
            cx.emit_lint(
                RENAMED_AND_REMOVED_LINTS,
                AttributeLintKind::RenamedLint {
                    name: *full_name,
                    replace: new_name,
                    suggestion: span,
                },
                span,
            );
            None
        }

        CheckLintNameResult::Removed(reason) => {
            cx.emit_lint(
                RENAMED_AND_REMOVED_LINTS,
                AttributeLintKind::RemovedLint { name: *full_name, reason },
                span,
            );
            None
        }

        CheckLintNameResult::NoLint(suggestion) => {
            cx.emit_lint(
                UNKNOWN_LINTS,
                AttributeLintKind::UnknownLint { name: *full_name, suggestion, span },
                span,
            );
            None
        }
    }
}
