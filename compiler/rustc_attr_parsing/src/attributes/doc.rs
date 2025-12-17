use rustc_ast::ast::{AttrStyle, LitKind, MetaItemLit};
use rustc_feature::template;
use rustc_hir::attrs::{
    AttributeKind, CfgEntry, CfgHideShow, CfgInfo, DocAttribute, DocInline, HideOrShow,
};
use rustc_hir::lints::AttributeLintKind;
use rustc_span::{Span, Symbol, edition, sym};
use thin_vec::ThinVec;

use super::prelude::{ALL_TARGETS, AllowedTargets};
use super::{AcceptMapping, AttributeParser};
use crate::context::{AcceptContext, FinalizeContext, Stage};
use crate::parser::{ArgParser, MetaItemOrLitParser, MetaItemParser, OwnedPathParser};
use crate::session_diagnostics::{
    DocAliasBadChar, DocAliasEmpty, DocAliasMalformed, DocAliasStartEnd, DocAttributeNotAttribute,
    DocKeywordNotKeyword,
};

fn check_keyword<S: Stage>(cx: &mut AcceptContext<'_, '_, S>, keyword: Symbol, span: Span) -> bool {
    // FIXME: Once rustdoc can handle URL conflicts on case insensitive file systems, we
    // can remove the `SelfTy` case here, remove `sym::SelfTy`, and update the
    // `#[doc(keyword = "SelfTy")` attribute in `library/std/src/keyword_docs.rs`.
    if keyword.is_reserved(|| edition::LATEST_STABLE_EDITION)
        || keyword.is_weak()
        || keyword == sym::SelfTy
    {
        return true;
    }
    cx.emit_err(DocKeywordNotKeyword { span, keyword });
    false
}

fn check_attribute<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    attribute: Symbol,
    span: Span,
) -> bool {
    // FIXME: This should support attributes with namespace like `diagnostic::do_not_recommend`.
    if rustc_feature::BUILTIN_ATTRIBUTE_MAP.contains_key(&attribute) {
        return true;
    }
    cx.emit_err(DocAttributeNotAttribute { span, attribute });
    false
}

fn parse_keyword_and_attribute<S, F>(
    cx: &mut AcceptContext<'_, '_, S>,
    path: &OwnedPathParser,
    args: &ArgParser,
    attr_value: &mut Option<(Symbol, Span)>,
    callback: F,
) where
    S: Stage,
    F: FnOnce(&mut AcceptContext<'_, '_, S>, Symbol, Span) -> bool,
{
    let Some(nv) = args.name_value() else {
        cx.expected_name_value(args.span().unwrap_or(path.span()), path.word_sym());
        return;
    };

    let Some(value) = nv.value_as_str() else {
        cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
        return;
    };

    if !callback(cx, value, nv.value_span) {
        return;
    }

    if attr_value.is_some() {
        cx.duplicate_key(path.span(), path.word_sym().unwrap());
        return;
    }

    *attr_value = Some((value, path.span()));
}

#[derive(Default, Debug)]
pub(crate) struct DocParser {
    attribute: DocAttribute,
    nb_doc_attrs: usize,
}

impl DocParser {
    fn parse_single_test_doc_attr_item<S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, '_, S>,
        mip: &MetaItemParser,
    ) {
        let path = mip.path();
        let args = mip.args();

        match path.word_sym() {
            Some(sym::no_crate_inject) => {
                if let Err(span) = args.no_args() {
                    cx.expected_no_args(span);
                    return;
                }

                if self.attribute.no_crate_inject.is_some() {
                    cx.duplicate_key(path.span(), sym::no_crate_inject);
                    return;
                }

                self.attribute.no_crate_inject = Some(path.span())
            }
            Some(sym::attr) => {
                let Some(list) = args.list() else {
                    cx.expected_list(cx.attr_span, args);
                    return;
                };

                // FIXME: convert list into a Vec of `AttributeKind` because current code is awful.
                for attr in list.mixed() {
                    self.attribute.test_attrs.push(attr.span());
                }
            }
            Some(name) => {
                cx.emit_lint(
                    rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                    AttributeLintKind::DocTestUnknown { name },
                    path.span(),
                );
            }
            None => {
                cx.emit_lint(
                    rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                    AttributeLintKind::DocTestLiteral,
                    path.span(),
                );
            }
        }
    }

    fn add_alias<S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, '_, S>,
        alias: Symbol,
        span: Span,
    ) {
        let attr_str = "`#[doc(alias = \"...\")]`";
        if alias == sym::empty {
            cx.emit_err(DocAliasEmpty { span, attr_str });
            return;
        }

        let alias_str = alias.as_str();
        if let Some(c) =
            alias_str.chars().find(|&c| c == '"' || c == '\'' || (c.is_whitespace() && c != ' '))
        {
            cx.emit_err(DocAliasBadChar { span, attr_str, char_: c });
            return;
        }
        if alias_str.starts_with(' ') || alias_str.ends_with(' ') {
            cx.emit_err(DocAliasStartEnd { span, attr_str });
            return;
        }

        if let Some(first_definition) = self.attribute.aliases.get(&alias).copied() {
            cx.emit_lint(
                rustc_session::lint::builtin::UNUSED_ATTRIBUTES,
                AttributeLintKind::DuplicateDocAlias { first_definition },
                span,
            );
        }

        self.attribute.aliases.insert(alias, span);
    }

    fn parse_alias<S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, '_, S>,
        path: &OwnedPathParser,
        args: &ArgParser,
    ) {
        match args {
            ArgParser::NoArgs => {
                cx.emit_err(DocAliasMalformed { span: args.span().unwrap_or(path.span()) });
            }
            ArgParser::List(list) => {
                for i in list.mixed() {
                    let Some(alias) = i.lit().and_then(|i| i.value_str()) else {
                        cx.expected_string_literal(i.span(), i.lit());
                        continue;
                    };

                    self.add_alias(cx, alias, i.span());
                }
            }
            ArgParser::NameValue(nv) => {
                let Some(alias) = nv.value_as_str() else {
                    cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
                    return;
                };
                self.add_alias(cx, alias, nv.value_span);
            }
        }
    }

    fn parse_inline<S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, '_, S>,
        path: &OwnedPathParser,
        args: &ArgParser,
        inline: DocInline,
    ) {
        if let Err(span) = args.no_args() {
            cx.expected_no_args(span);
            return;
        }

        self.attribute.inline.push((inline, path.span()));
    }

    fn parse_cfg<S: Stage>(&mut self, cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) {
        // This function replaces cases like `cfg(all())` with `true`.
        fn simplify_cfg(cfg_entry: &mut CfgEntry) {
            match cfg_entry {
                CfgEntry::All(cfgs, span) if cfgs.is_empty() => {
                    *cfg_entry = CfgEntry::Bool(true, *span)
                }
                CfgEntry::Any(cfgs, span) if cfgs.is_empty() => {
                    *cfg_entry = CfgEntry::Bool(false, *span)
                }
                CfgEntry::Not(cfg, _) => simplify_cfg(cfg),
                _ => {}
            }
        }
        if let Some(mut cfg_entry) = super::cfg::parse_cfg(cx, args) {
            simplify_cfg(&mut cfg_entry);
            self.attribute.cfg.push(cfg_entry);
        }
    }

    fn parse_auto_cfg<S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, '_, S>,
        path: &OwnedPathParser,
        args: &ArgParser,
    ) {
        match args {
            ArgParser::NoArgs => {
                self.attribute.auto_cfg_change.push((true, path.span()));
            }
            ArgParser::List(list) => {
                for meta in list.mixed() {
                    let MetaItemOrLitParser::MetaItemParser(item) = meta else {
                        cx.emit_lint(
                            rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                            AttributeLintKind::DocAutoCfgExpectsHideOrShow,
                            meta.span(),
                        );
                        continue;
                    };
                    let (kind, attr_name) = match item.path().word_sym() {
                        Some(sym::hide) => (HideOrShow::Hide, sym::hide),
                        Some(sym::show) => (HideOrShow::Show, sym::show),
                        _ => {
                            cx.emit_lint(
                                rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                                AttributeLintKind::DocAutoCfgExpectsHideOrShow,
                                item.span(),
                            );
                            continue;
                        }
                    };
                    let ArgParser::List(list) = item.args() else {
                        cx.emit_lint(
                            rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                            AttributeLintKind::DocAutoCfgHideShowExpectsList { attr_name },
                            item.span(),
                        );
                        continue;
                    };

                    let mut cfg_hide_show = CfgHideShow { kind, values: ThinVec::new() };

                    for item in list.mixed() {
                        let MetaItemOrLitParser::MetaItemParser(sub_item) = item else {
                            cx.emit_lint(
                                rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                                AttributeLintKind::DocAutoCfgHideShowUnexpectedItem { attr_name },
                                item.span(),
                            );
                            continue;
                        };
                        match sub_item.args() {
                            a @ (ArgParser::NoArgs | ArgParser::NameValue(_)) => {
                                let Some(name) = sub_item.path().word_sym() else {
                                    cx.expected_identifier(sub_item.path().span());
                                    continue;
                                };
                                if let Ok(CfgEntry::NameValue { name, value, .. }) =
                                    super::cfg::parse_name_value(
                                        name,
                                        sub_item.path().span(),
                                        a.name_value(),
                                        sub_item.span(),
                                        cx,
                                    )
                                {
                                    cfg_hide_show.values.push(CfgInfo {
                                        name,
                                        name_span: sub_item.path().span(),
                                        // If `value` is `Some`, `a.name_value()` will always return
                                        // `Some` as well.
                                        value: value
                                            .map(|v| (v, a.name_value().unwrap().value_span)),
                                    })
                                }
                            }
                            _ => {
                                cx.emit_lint(
                                    rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                                    AttributeLintKind::DocAutoCfgHideShowUnexpectedItem {
                                        attr_name,
                                    },
                                    sub_item.span(),
                                );
                                continue;
                            }
                        }
                    }
                    self.attribute.auto_cfg.push((cfg_hide_show, path.span()));
                }
            }
            ArgParser::NameValue(nv) => {
                let MetaItemLit { kind: LitKind::Bool(bool_value), span, .. } = nv.value_as_lit()
                else {
                    cx.emit_lint(
                        rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                        AttributeLintKind::DocAutoCfgWrongLiteral,
                        nv.value_span,
                    );
                    return;
                };
                self.attribute.auto_cfg_change.push((*bool_value, *span));
            }
        }
    }

    fn parse_single_doc_attr_item<S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, '_, S>,
        mip: &MetaItemParser,
    ) {
        let path = mip.path();
        let args = mip.args();

        macro_rules! no_args {
            ($ident: ident) => {{
                if let Err(span) = args.no_args() {
                    cx.expected_no_args(span);
                    return;
                }

                // FIXME: It's errorring when the attribute is passed multiple times on the command
                // line.
                // The right fix for this would be to only check this rule if the attribute is
                // not set on the command line but directly in the code.
                // if self.attribute.$ident.is_some() {
                //     cx.duplicate_key(path.span(), path.word_sym().unwrap());
                //     return;
                // }

                self.attribute.$ident = Some(path.span());
            }};
        }
        macro_rules! string_arg {
            ($ident: ident) => {{
                let Some(nv) = args.name_value() else {
                    cx.expected_name_value(args.span().unwrap_or(path.span()), path.word_sym());
                    return;
                };

                let Some(s) = nv.value_as_str() else {
                    cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
                    return;
                };

                // FIXME: It's errorring when the attribute is passed multiple times on the command
                // line.
                // The right fix for this would be to only check this rule if the attribute is
                // not set on the command line but directly in the code.
                // if self.attribute.$ident.is_some() {
                //     cx.duplicate_key(path.span(), path.word_sym().unwrap());
                //     return;
                // }

                self.attribute.$ident = Some((s, path.span()));
            }};
        }

        match path.word_sym() {
            Some(sym::alias) => self.parse_alias(cx, path, args),
            Some(sym::hidden) => no_args!(hidden),
            Some(sym::html_favicon_url) => string_arg!(html_favicon_url),
            Some(sym::html_logo_url) => string_arg!(html_logo_url),
            Some(sym::html_no_source) => no_args!(html_no_source),
            Some(sym::html_playground_url) => string_arg!(html_playground_url),
            Some(sym::html_root_url) => string_arg!(html_root_url),
            Some(sym::issue_tracker_base_url) => string_arg!(issue_tracker_base_url),
            Some(sym::inline) => self.parse_inline(cx, path, args, DocInline::Inline),
            Some(sym::no_inline) => self.parse_inline(cx, path, args, DocInline::NoInline),
            Some(sym::masked) => no_args!(masked),
            Some(sym::cfg) => self.parse_cfg(cx, args),
            Some(sym::notable_trait) => no_args!(notable_trait),
            Some(sym::keyword) => parse_keyword_and_attribute(
                cx,
                path,
                args,
                &mut self.attribute.keyword,
                check_keyword,
            ),
            Some(sym::attribute) => parse_keyword_and_attribute(
                cx,
                path,
                args,
                &mut self.attribute.attribute,
                check_attribute,
            ),
            Some(sym::fake_variadic) => no_args!(fake_variadic),
            Some(sym::search_unbox) => no_args!(search_unbox),
            Some(sym::rust_logo) => no_args!(rust_logo),
            Some(sym::auto_cfg) => self.parse_auto_cfg(cx, path, args),
            Some(sym::test) => {
                let Some(list) = args.list() else {
                    cx.emit_lint(
                        rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                        AttributeLintKind::DocTestTakesList,
                        args.span().unwrap_or(path.span()),
                    );
                    return;
                };

                for i in list.mixed() {
                    match i {
                        MetaItemOrLitParser::MetaItemParser(mip) => {
                            self.parse_single_test_doc_attr_item(cx, mip);
                        }
                        MetaItemOrLitParser::Lit(lit) => {
                            cx.unexpected_literal(lit.span);
                        }
                        MetaItemOrLitParser::Err(..) => {
                            // already had an error here, move on.
                        }
                    }
                }
            }
            Some(sym::spotlight) => {
                cx.emit_lint(
                    rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                    AttributeLintKind::DocUnknownSpotlight { span: path.span() },
                    path.span(),
                );
            }
            Some(sym::include) if let Some(nv) = args.name_value() => {
                let inner = match cx.attr_style {
                    AttrStyle::Outer => "",
                    AttrStyle::Inner => "!",
                };
                cx.emit_lint(
                    rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                    AttributeLintKind::DocUnknownInclude {
                        inner,
                        value: nv.value_as_lit().symbol,
                        span: path.span(),
                    },
                    path.span(),
                );
            }
            Some(name @ (sym::passes | sym::no_default_passes)) => {
                cx.emit_lint(
                    rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                    AttributeLintKind::DocUnknownPasses { name, span: path.span() },
                    path.span(),
                );
            }
            Some(sym::plugins) => {
                cx.emit_lint(
                    rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                    AttributeLintKind::DocUnknownPlugins { span: path.span() },
                    path.span(),
                );
            }
            Some(name) => {
                cx.emit_lint(
                    rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                    AttributeLintKind::DocUnknownAny { name },
                    path.span(),
                );
            }
            None => {
                let full_name =
                    path.segments().map(|s| s.as_str()).intersperse("::").collect::<String>();
                cx.emit_lint(
                    rustc_session::lint::builtin::INVALID_DOC_ATTRIBUTES,
                    AttributeLintKind::DocUnknownAny { name: Symbol::intern(&full_name) },
                    path.span(),
                );
            }
        }
    }

    fn accept_single_doc_attr<S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) {
        match args {
            ArgParser::NoArgs => {
                let suggestions = cx.suggestions();
                let span = cx.attr_span;
                cx.emit_lint(
                    rustc_session::lint::builtin::ILL_FORMED_ATTRIBUTE_INPUT,
                    AttributeLintKind::IllFormedAttributeInput { suggestions, docs: None },
                    span,
                );
            }
            ArgParser::List(items) => {
                for i in items.mixed() {
                    match i {
                        MetaItemOrLitParser::MetaItemParser(mip) => {
                            self.nb_doc_attrs += 1;
                            self.parse_single_doc_attr_item(cx, mip);
                        }
                        MetaItemOrLitParser::Lit(lit) => {
                            cx.expected_name_value(lit.span, None);
                        }
                        MetaItemOrLitParser::Err(..) => {
                            // already had an error here, move on.
                        }
                    }
                }
            }
            ArgParser::NameValue(nv) => {
                if nv.value_as_str().is_none() {
                    cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
                } else {
                    unreachable!(
                        "Should have been handled at the same time as sugar-syntaxed doc comments"
                    );
                }
            }
        }
    }
}

impl<S: Stage> AttributeParser<S> for DocParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::doc],
        template!(
            List: &[
                "alias",
                "attribute",
                "hidden",
                "html_favicon_url",
                "html_logo_url",
                "html_no_source",
                "html_playground_url",
                "html_root_url",
                "issue_tracker_base_url",
                "inline",
                "no_inline",
                "masked",
                "cfg",
                "notable_trait",
                "keyword",
                "fake_variadic",
                "search_unbox",
                "rust_logo",
                "auto_cfg",
                "test",
                "spotlight",
                "include",
                "no_default_passes",
                "passes",
                "plugins",
            ],
            NameValueStr: "string"
        ),
        |this, cx, args| {
            this.accept_single_doc_attr(cx, args);
        },
    )];
    // FIXME: Currently emitted from 2 different places, generating duplicated warnings.
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
    // const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
    //     Allow(Target::ExternCrate),
    //     Allow(Target::Use),
    //     Allow(Target::Static),
    //     Allow(Target::Const),
    //     Allow(Target::Fn),
    //     Allow(Target::Mod),
    //     Allow(Target::ForeignMod),
    //     Allow(Target::TyAlias),
    //     Allow(Target::Enum),
    //     Allow(Target::Variant),
    //     Allow(Target::Struct),
    //     Allow(Target::Field),
    //     Allow(Target::Union),
    //     Allow(Target::Trait),
    //     Allow(Target::TraitAlias),
    //     Allow(Target::Impl { of_trait: true }),
    //     Allow(Target::Impl { of_trait: false }),
    //     Allow(Target::AssocConst),
    //     Allow(Target::Method(MethodKind::Inherent)),
    //     Allow(Target::Method(MethodKind::Trait { body: true })),
    //     Allow(Target::Method(MethodKind::Trait { body: false })),
    //     Allow(Target::Method(MethodKind::TraitImpl)),
    //     Allow(Target::AssocTy),
    //     Allow(Target::ForeignFn),
    //     Allow(Target::ForeignStatic),
    //     Allow(Target::ForeignTy),
    //     Allow(Target::MacroDef),
    //     Allow(Target::Crate),
    //     Error(Target::WherePredicate),
    // ]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if self.nb_doc_attrs != 0 {
            Some(AttributeKind::Doc(Box::new(self.attribute)))
        } else {
            None
        }
    }
}
