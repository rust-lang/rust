use rustc_ast::join_path_syms;
use rustc_attr_ir::AttributeKind;
use rustc_attr_ir::lint::{LintCheck, LintCheckKind};
use rustc_attr_ir::target::{AssocCtxt, MethodKind, Target};
use rustc_lint_defs::builtin::{RENAMED_AND_REMOVED_LINTS, UNKNOWN_LINTS, UNUSED_ATTRIBUTES};
use rustc_lint_defs::{CheckLintNameResult, LintId};
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, sym};
use thin_vec::ThinVec;

use crate::attributes::{AcceptMapping, AttributeParser, AttributeStability};
use crate::context::{AcceptContext, ExpectStringLiteral, FinalizeContext};
use crate::diagnostics::{
    DeprecatedLintName, RemovedLint, RenamedLint, RenamedLintSuggestion, UnknownLint,
    UnknownLintSuggestion, UnknownToolInScopedLint,
};
use crate::parser::{ArgParser, MetaItemOrLitParser};
use crate::target_checking::AllowedTargets;
use crate::target_checking::Policy::{Allow, Warn};
use crate::{AttributeTemplate, diagnostics, template};

const LINT_TEMPLATE: AttributeTemplate = template!(
    List: &["lint1", "lint1, lint2, ...", r#"lint1, lint2, lint3, reason = "...""#],
    "https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes"
);

#[derive(Default, Debug)]
pub(crate) struct LintParser {
    lints: ThinVec<LintCheck>,
}

impl LintParser {
    fn parse(&mut self, kind: LintCheckKind, cx: &mut AcceptContext<'_, '_>, args: &ArgParser) {
        let attr_span = cx.attr_span;
        let attr_id = cx.attr_id.expect("no `AttrId` for lint attribute");

        if let Some(list) = cx.expect_list(args, cx.attr_span) {
            let mut parsers = list.sub_parsers();
            // Optionally, the last (and only the last)
            // element can be `reason = "reason"`
            let reason = try {
                let p = parsers.last()?.meta_item()?;
                let nv = p.args().as_name_value()?;
                if !p.path().word_is(sym::reason) {
                    cx.emit_err(diagnostics::MalformedAttribute {
                        span: p.span(),
                        kind: diagnostics::MalformedAttributeKind::BadAttributeArgument(p.span()),
                    });
                } else {
                    parsers = &parsers[..(parsers.len() - 1)];
                }

                let reason = nv.expect_string_literal(cx)?;
                reason
            };

            for item in parsers {
                match item {
                    MetaItemOrLitParser::MetaItemParser(p) => {
                        match p.args() {
                            ArgParser::NoArgs => {
                                let (tool, lint_name) = match &*p.path().0.segments {
                                    [] => unreachable!(),
                                    [lint_name] => (None, lint_name.ident.name),
                                    [tool, lint_name] => (Some(tool.ident), lint_name.ident.name),
                                    // Not reachable for any builtin lints,
                                    // but reachable using `register_tool`.
                                    [tool, multi_segment_lint_name @ ..] => (
                                        Some(tool.ident),
                                        Symbol::intern(&join_path_syms(
                                            multi_segment_lint_name
                                                .iter()
                                                .map(|seg| seg.ident.name),
                                        )),
                                    ),
                                };

                                let lint_name_normalized = if let Some((_lint_ids, new_lint_name)) =
                                    LintParser::check_lint_name(lint_name, tool, p.span(), cx)
                                {
                                    new_lint_name
                                } else {
                                    // Discard lint attributes with unknown lints in them
                                    continue;
                                };

                                self.lints.push(LintCheck {
                                    tool: tool.map(|tool| tool.name),
                                    lint_name_normalized,
                                    lint_span: p.span(),
                                    kind,
                                    attr_id,
                                    reason,
                                    attr_span,
                                })
                            }
                            // We've found a `reason = "reason"` but we're not the last element.
                            ArgParser::NameValue(nv) if p.path().word_is(sym::reason) => {
                                cx.emit_err(diagnostics::MalformedAttribute {
                                    span: p.span(),
                                    kind: diagnostics::MalformedAttributeKind::ReasonMustComeLast(
                                        item.span(),
                                    ),
                                });
                                nv.expect_string_literal(cx);
                            }
                            ArgParser::NameValue(_) | ArgParser::List(_) => {
                                cx.emit_err(diagnostics::MalformedAttribute {
                                    span: p.span(),
                                    kind: diagnostics::MalformedAttributeKind::BadAttributeArgument(
                                        item.span(),
                                    ),
                                });
                            }
                        }
                    }
                    MetaItemOrLitParser::Lit(lit) => {
                        cx.emit_err(diagnostics::BadAttributeLit {
                            span: item.span(),
                            help: if lit.kind.is_str() {
                                diagnostics::BadAttributeLitHelp::StrLit {
                                    sym: lit.symbol,
                                    kind: kind.sym(),
                                }
                            } else {
                                diagnostics::BadAttributeLitHelp::Other { kind: kind.sym() }
                            },
                        });
                    }
                }
            }
            if parsers.is_empty() {
                cx.emit_lint(
                    UNUSED_ATTRIBUTES,
                    diagnostics::Unused {
                        attr_span,
                        note: if list.is_empty() {
                            diagnostics::UnusedNote::EmptyList { name: kind.sym() }
                        } else {
                            diagnostics::UnusedNote::NoLints { name: kind.sym() }
                        },
                    },
                    attr_span,
                );
            }
        }
    }

    fn check_lint_name<'a>(
        lint_name: Symbol,
        tool: Option<Ident>,
        lint_span: Span,
        cx: &mut AcceptContext<'_, 'a>,
    ) -> Option<(&'a [LintId], Symbol)> {
        let lint_result = cx.sess.lint_store.as_ref().unwrap().check_lint_name(
            lint_name.as_str(),
            tool.map(|tool| tool.name),
            cx.lint_tools.unwrap(),
        );

        // FIXME(164898) Lint attributes are parsed twice, once on the AST and once on the HIR
        // Both times emit diagnostics, for historic reasons
        // We only want to emit these lints once, on the HIR.
        // On the AST `target_span` will be `DUMMY_SP`, which we use to detect which instance we are.
        // #164898 will fix this, after that PR this check can be removed
        let should_lint = cx.target_span != DUMMY_SP;

        let (ids, name) = match lint_result {
            CheckLintNameResult::Ok(ids) => (ids, lint_name),

            CheckLintNameResult::Tool(ids, new_lint_name) => {
                let name = match new_lint_name {
                    None => {
                        let complete_name = &format!("{}::{lint_name}", tool.unwrap().name);
                        Symbol::intern(complete_name)
                    }
                    Some(new_lint_name) => {
                        let symbol = Symbol::intern(&new_lint_name);
                        if should_lint {
                            cx.emit_lint(
                                RENAMED_AND_REMOVED_LINTS,
                                DeprecatedLintName {
                                    name: lint_name.to_string(),
                                    suggestion: lint_span,
                                    replace: new_lint_name,
                                },
                                lint_span,
                            );
                        }
                        symbol
                    }
                };
                (ids, name)
            }

            CheckLintNameResult::MissingTool => {
                // If `MissingTool` is returned, then either the lint does not
                // exist in the tool or the code was not compiled with the tool and
                // therefore the lint was never added to the `LintStore`. To detect
                // this is the responsibility of the lint tool.
                return None;
            }

            CheckLintNameResult::NoTool => {
                if should_lint {
                    cx.emit_err(UnknownToolInScopedLint {
                        span: Some(tool.unwrap().span),
                        tool_name: tool.unwrap().name,
                        lint_name: lint_name.to_string(),
                        is_nightly_build: cx.sess.is_nightly_build(),
                    });
                }
                return None;
            }

            CheckLintNameResult::Renamed(ref replace) => {
                if should_lint {
                    let suggestion = RenamedLintSuggestion::WithSpan {
                        suggestion: lint_span,
                        replace: replace.clone(),
                    };
                    let name = tool
                        .map(|Ident { name, .. }| format!("{name}::{lint_name}"))
                        .unwrap_or_else(|| lint_name.to_string());
                    cx.emit_lint(
                        RENAMED_AND_REMOVED_LINTS,
                        RenamedLint { name, replace: replace.clone(), suggestion },
                        lint_span,
                    );
                }

                // If this lint was renamed, apply the new lint instead of ignoring the
                // attribute. Ignore any errors or warnings that happen because the new
                // name is inaccurate.
                // NOTE: `new_name` already includes the tool name, so we don't
                // have to add it again.
                let CheckLintNameResult::Ok(ids) = cx
                    .sess
                    .lint_store
                    .as_ref()
                    .unwrap()
                    .check_lint_name(replace, None, cx.lint_tools.unwrap())
                else {
                    panic!("renamed lint does not exist: {replace}");
                };

                (ids, Symbol::intern(&replace))
            }

            CheckLintNameResult::Removed(ref reason) => {
                if should_lint {
                    let name = if let Some(Ident { name, .. }) = tool {
                        format!("{name}::{lint_name}")
                    } else {
                        lint_name.to_string()
                    };
                    cx.emit_lint(
                        RENAMED_AND_REMOVED_LINTS,
                        RemovedLint { name, reason: reason.clone() },
                        lint_span,
                    );
                }
                return None;
            }

            CheckLintNameResult::NoLint(suggestion) => {
                if should_lint {
                    let name = tool
                        .map(|Ident { name, .. }| format!("{name}::{lint_name}"))
                        .unwrap_or_else(|| lint_name.to_string());
                    let suggestion =
                        suggestion.map(|(replace, from_rustc)| UnknownLintSuggestion::WithSpan {
                            suggestion: lint_span,
                            replace,
                            from_rustc,
                        });
                    cx.emit_lint(UNKNOWN_LINTS, UnknownLint { name, suggestion }, lint_span);
                }
                return None;
            }
        };
        Some((ids, name))
    }
}

impl AttributeParser for LintParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[
        (&[sym::allow], LINT_TEMPLATE, AttributeStability::Stable, |this, cx, args| {
            this.parse(LintCheckKind::Allow, cx, args)
        }),
        (&[sym::warn], LINT_TEMPLATE, AttributeStability::Stable, |this, cx, args| {
            this.parse(LintCheckKind::Warn, cx, args)
        }),
        (&[sym::deny], LINT_TEMPLATE, AttributeStability::Stable, |this, cx, args| {
            this.parse(LintCheckKind::Deny, cx, args)
        }),
        (&[sym::forbid], LINT_TEMPLATE, AttributeStability::Stable, |this, cx, args| {
            this.parse(LintCheckKind::Forbid, cx, args)
        }),
        (&[sym::expect], LINT_TEMPLATE, AttributeStability::Stable, |this, cx, args| {
            this.parse(LintCheckKind::Expect, cx, args)
        }),
    ];
    const ALLOWED_TARGETS: AllowedTargets<'_> = {
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
            Allow(Target::AssocConst(AssocCtxt::Impl { of_trait: false })),
            Allow(Target::AssocConst(AssocCtxt::Trait)),
            Allow(Target::AssocConst(AssocCtxt::Impl { of_trait: true })),
            Allow(Target::Method(MethodKind::Inherent)),
            Allow(Target::Method(MethodKind::Trait { body: false })),
            Allow(Target::Method(MethodKind::Trait { body: true })),
            Allow(Target::Method(MethodKind::TraitImpl)),
            Allow(Target::AssocTy(AssocCtxt::Impl { of_trait: false })),
            Allow(Target::AssocTy(AssocCtxt::Trait)),
            Allow(Target::AssocTy(AssocCtxt::Impl { of_trait: true })),
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
            Allow(Target::ConstParam),
            Allow(Target::LifetimeParam),
            Allow(Target::TypeParam),
            Allow(Target::Loop),
            Allow(Target::ForLoop),
            Allow(Target::While),
            Allow(Target::Break),
            Warn(Target::MacroCall),
        ])
    };
    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        if self.lints.is_empty() { None } else { Some(AttributeKind::LintCheck(self.lints)) }
    }
}
