use rustc_attr_ir::AttributeKind;
use rustc_attr_ir::lint::{LintCheck, LintCheckKind};
use rustc_attr_ir::target::{AssocCtxt, MethodKind, Target};
use rustc_lint_defs::builtin::UNUSED_ATTRIBUTES;
use rustc_span::{Span, Symbol, sym};
use thin_vec::ThinVec;

use crate::attributes::{AcceptMapping, AttributeParser, AttributeStability};
use crate::context::{AcceptContext, ExpectStringLiteral, FinalizeContext};
use crate::parser::ArgParser;
use crate::target_checking::AllowedTargets;
use crate::target_checking::Policy::{Allow, Warn};
use crate::{AttributeTemplate, diagnostics, template};

const LINT_TEMPLATE: AttributeTemplate = template!(
    List: &["lint1", "lint1, lint2, ...", r#"lint1, lint2, lint3, reason = "...""#],
    "https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes"
);

#[derive(Default, Debug)]
pub(crate) struct LintParser {
    attr_index: u16,
    lints: ThinVec<LintCheck>,
}

impl LintParser {
    fn parse(&mut self, kind: LintCheckKind, cx: &mut AcceptContext<'_, '_>, args: &ArgParser) {
        let attr_index = self.attr_index;
        let attr_span = cx.attr_span;
        let attr_id = cx.attr_id.expect("no `AttrId` for lint attribute");
        let mut lints: Vec<(ThinVec<Symbol>, Span)> = Vec::new();

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
                        sub: diagnostics::MalformedAttributeSub::BadAttributeArgument(p.span()),
                    });
                } else {
                    parsers = &parsers[..(parsers.len() - 1)];
                }

                let reason = nv.expect_string_literal(cx)?;
                reason
            };

            for item in parsers {
                if let Some(p) = item.meta_item() {
                    match p.args() {
                        ArgParser::NoArgs => {
                            lints.push((p.path().segments().map(|i| i.name).collect(), p.span()))
                        }
                        // We're found a `reason = "reason"` but we're not the last element.
                        ArgParser::NameValue(nv) if p.path().word_is(sym::reason) => {
                            cx.emit_err(diagnostics::MalformedAttribute {
                                span: p.span(),
                                sub: diagnostics::MalformedAttributeSub::ReasonMustComeLast(
                                    item.span(),
                                ),
                            });
                            nv.expect_string_literal(cx);
                        }
                        ArgParser::NameValue(_) | ArgParser::List(_) => {
                            cx.emit_err(diagnostics::MalformedAttribute {
                                span: p.span(),
                                sub: diagnostics::MalformedAttributeSub::BadAttributeArgument(
                                    item.span(),
                                ),
                            });
                        }
                    }
                } else {
                    cx.emit_err(diagnostics::MalformedAttribute {
                        span: item.span(),
                        sub: diagnostics::MalformedAttributeSub::BadAttributeArgument(item.span()),
                    });
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

            for (lint_index, (name, span)) in lints.into_iter().enumerate() {
                self.lints.push(LintCheck {
                    name,
                    span,
                    lint_index: lint_index as u16,
                    attr_index,
                    kind,
                    attr_id,
                    reason,
                    attr_span,
                })
            }
        }

        self.attr_index += 1
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
