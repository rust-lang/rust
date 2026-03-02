use std::marker::PhantomData;

use rustc_ast::LitKind;
use rustc_hir::HashIgnoredAttrId;
use rustc_hir::attrs::{LintAttribute, LintAttributeKind, LintInstance};
use rustc_hir::lints::AttributeLintKind;
use rustc_hir::target::GenericParamKind;
use rustc_middle::bug;
use rustc_session::lint::CheckLintNameResult;
use rustc_session::lint::builtin::{RENAMED_AND_REMOVED_LINTS, UNKNOWN_LINTS};

use super::prelude::*;
use crate::session_diagnostics::UnknownToolInScopedLint;

pub(crate) trait Lint {
    const KIND: LintAttributeKind;
    const ATTR_SYMBOL: Symbol;
}
#[derive(Default)]
pub(crate) struct Expect;
pub(crate) type ExpectParser = LintParser<Expect>;

impl Lint for Expect {
    const KIND: LintAttributeKind = LintAttributeKind::Expect;
    const ATTR_SYMBOL: Symbol = sym::expect;
}

#[derive(Default)]
pub(crate) struct Allow;
pub(crate) type AllowParser = LintParser<Allow>;

impl Lint for Allow {
    const KIND: LintAttributeKind = LintAttributeKind::Allow;
    const ATTR_SYMBOL: Symbol = sym::allow;
}
#[derive(Default)]
pub(crate) struct Warn;
pub(crate) type WarnParser = LintParser<Warn>;

impl Lint for Warn {
    const KIND: LintAttributeKind = LintAttributeKind::Warn;
    const ATTR_SYMBOL: Symbol = sym::warn;
}

#[derive(Default)]
pub(crate) struct Deny;
pub(crate) type DenyParser = LintParser<Deny>;

impl Lint for Deny {
    const KIND: LintAttributeKind = LintAttributeKind::Deny;
    const ATTR_SYMBOL: Symbol = sym::deny;
}

#[derive(Default)]
pub(crate) struct Forbid;
pub(crate) type ForbidParser = LintParser<Forbid>;

impl Lint for Forbid {
    const KIND: LintAttributeKind = LintAttributeKind::Forbid;
    const ATTR_SYMBOL: Symbol = sym::forbid;
}

#[derive(Default)]
pub(crate) struct LintParser<T: Lint> {
    lint_attrs: ThinVec<LintAttribute>,
    attr_index: usize,
    phantom: PhantomData<T>,
}

impl<S: Stage, T: Lint + 'static + Default> AttributeParser<S> for LintParser<T> {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[T::ATTR_SYMBOL],
        template!(
            List: &["lint1", "lint1, lint2, ...", r#"lint1, lint2, lint3, reason = "...""#],
            "https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes"
        ),
        |this, cx, args| {
            if let Some(lint_attr) = validate_lint_attr(cx, args, T::ATTR_SYMBOL, this.attr_index) {
                this.lint_attrs.push(lint_attr);
            }
            this.attr_index += 1;
        },
    )];

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
            Warn(Target::MacroCall),
        ])
    };

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if !self.lint_attrs.is_empty() {
            Some(AttributeKind::LintAttribute { sub_attrs: self.lint_attrs, kind: T::KIND })
        } else {
            None
        }
    }
}

#[inline(always)]
fn validate_lint_attr<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    args: &ArgParser,
    attr_name: Symbol,
    attr_index: usize,
) -> Option<LintAttribute> {
    let Some(list) = args.list() else {
        cx.expected_list(cx.inner_span, args);
        return None;
    };
    let mut list = list.mixed().peekable();

    let mut skip_reason_check = false;
    let mut errored = false;
    let mut reason = None;
    let mut lint_instances = ThinVec::new();
    let mut lint_index = 0;

    while let Some(item) = list.next() {
        let Some(meta_item) = item.meta_item() else {
            cx.expected_identifier(item.span());
            errored = true;
            continue;
        };

        match meta_item.args() {
            ArgParser::NameValue(nv_parser) if meta_item.path().word_is(sym::reason) => {
                //FIXME replace this with duplicate check?
                if list.peek().is_some() {
                    cx.expected_nv_as_last_argument(meta_item.span(), attr_name, sym::reason);
                    errored = true;
                    continue;
                }

                let val_lit = nv_parser.value_as_lit();
                let LitKind::Str(reason_sym, _) = val_lit.kind else {
                    cx.expected_string_literal(nv_parser.value_span, Some(val_lit));
                    errored = true;
                    continue;
                };
                reason = Some(reason_sym);
            }
            ArgParser::NameValue(_) => {
                cx.expected_specific_argument(meta_item.span(), &[sym::reason]);
                errored = true;
            }
            ArgParser::List(list) => {
                cx.expected_no_args(list.span);
                errored = true;
            }
            ArgParser::NoArgs => {
                let mut segments = meta_item.path().segments();

                let Some(tool_or_name) = segments.next() else {
                    bug!("first segment should always exist");
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

                if let Some(ids) = check_lint(
                    cx,
                    Symbol::intern(&name),
                    tool_name,
                    tool_span,
                    meta_item.span(),
                    lint_index,
                ) {
                    lint_instances.extend(ids);
                    skip_reason_check = true
                } else {
                    skip_reason_check = true
                }
                lint_index += 1;
            }
        }
    }
    if !skip_reason_check && !errored && lint_instances.is_empty() {
        cx.warn_empty_attribute(cx.attr_span);
    }

    (!errored).then_some(LintAttribute {
        reason,
        lint_instances,
        attr_span: cx.attr_span,
        attr_style: cx.attr_style,
        attr_id: HashIgnoredAttrId { attr_id: cx.attr_id },
        attr_index,
    })
}

fn check_lint<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    original_name: Symbol,
    tool_name: Option<Symbol>,
    tool_span: Option<Span>,
    span: Span,
    lint_index: usize,
) -> Option<Vec<LintInstance>> {
    let Some(lint_store) = &cx.sess.lint_store else {
        bug!("lint_store required while parsing attributes");
    };
    let Some(tools) = cx.tools else {
        bug!("tools required while parsing attributes");
    };
    if tools.is_empty() {
        bug!("tools should never be empty")
    }
    let full_name = tool_name
        .map(|tool| Symbol::intern(&format!("{tool}::{}", &original_name)))
        .unwrap_or(original_name);
    let mut lint_ids = Vec::new();
    match lint_store.check_lint_name(original_name.as_str(), tool_name, tools) {
        CheckLintNameResult::Ok(ids) => {
            for id in ids {
                lint_ids.push(LintInstance::new(full_name, id.to_string(), span, lint_index));
            }
        }
        CheckLintNameResult::Tool(ids, new_lint_name) => {
            let _name = match new_lint_name {
                None => original_name,
                Some(new_lint_name) => {
                    let new_lint_name = Symbol::intern(&new_lint_name);
                    cx.emit_lint(
                        RENAMED_AND_REMOVED_LINTS,
                        AttributeLintKind::DeprecatedLintName {
                            name: full_name,
                            suggestion: span,
                            replace: new_lint_name,
                        },
                        span,
                    );
                    new_lint_name
                }
            };
            for id in ids {
                lint_ids.push(LintInstance::new(full_name, id.to_string(), span, lint_index));
            }
        }

        CheckLintNameResult::MissingTool => {
            // If `MissingTool` is returned, then either the lint does not
            // exist in the tool or the code was not compiled with the tool and
            // therefore the lint was never added to the `LintStore`. To detect
            // this is the responsibility of the lint tool.
            return None;
        }

        CheckLintNameResult::NoTool => {
            cx.emit_err(UnknownToolInScopedLint {
                span: tool_span,
                tool_name: tool_name.unwrap(),
                full_lint_name: full_name,
                is_nightly_build: cx.sess.is_nightly_build(),
            });
            return None;
        }

        CheckLintNameResult::Renamed(replace) => {
            cx.emit_lint(
                RENAMED_AND_REMOVED_LINTS,
                AttributeLintKind::RenamedLint { name: full_name, replace, suggestion: span },
                span,
            );

            // If this lint was renamed, apply the new lint instead of ignoring the
            // attribute. Ignore any errors or warnings that happen because the new
            // name is inaccurate.
            // NOTE: `new_name` already includes the tool name, so we don't
            // have to add it again.
            let ids = match lint_store.check_lint_name(replace.as_str(), None, tools) {
                CheckLintNameResult::Ok(ids) => ids,
                _ => panic!("renamed lint does not exist: {replace}"),
            };

            for id in ids {
                lint_ids.push(LintInstance::new(full_name, id.to_string(), span, lint_index));
            }
        }

        CheckLintNameResult::RenamedToolLint(new_name) => {
            cx.emit_lint(
                RENAMED_AND_REMOVED_LINTS,
                AttributeLintKind::RenamedLint {
                    name: full_name,
                    replace: new_name,
                    suggestion: span,
                },
                span,
            );
            return None;
        }

        CheckLintNameResult::Removed(reason) => {
            cx.emit_lint(
                RENAMED_AND_REMOVED_LINTS,
                AttributeLintKind::RemovedLint { name: full_name, reason },
                span,
            );
            return None;
        }

        CheckLintNameResult::NoLint(suggestion) => {
            cx.emit_lint(
                UNKNOWN_LINTS,
                AttributeLintKind::UnknownLint { name: full_name, suggestion, span },
                span,
            );
            return None;
        }
    }
    Some(lint_ids)
}
