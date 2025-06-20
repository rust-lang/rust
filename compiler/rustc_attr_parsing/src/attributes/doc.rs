use rustc_attr_data_structures::lints::AttributeLintKind;
use rustc_attr_data_structures::{AttributeKind, DocAttribute, DocInline};
use rustc_errors::MultiSpan;
use rustc_feature::template;
use rustc_span::{Span, Symbol, edition, sym};

use super::{AcceptMapping, AttributeParser};
use crate::context::{AcceptContext, FinalizeContext, Stage};
use crate::fluent_generated as fluent;
use crate::parser::{ArgParser, MetaItemOrLitParser, MetaItemParser, PathParser};
use crate::session_diagnostics::{
    DocAliasBadChar, DocAliasEmpty, DocAliasStartEnd, DocKeywordConflict, DocKeywordNotKeyword,
};

#[derive(Default)]
pub(crate) struct DocParser {
    attribute: DocAttribute,
}

impl DocParser {
    fn parse_single_test_doc_attr_item<'c, S: Stage>(
        &mut self,
        cx: &'c mut AcceptContext<'_, '_, S>,
        mip: &'c MetaItemParser<'_>,
    ) {
        let path = mip.path();
        let args = mip.args();

        match path.word_sym() {
            Some(sym::no_crate_inject) => {
                if !args.no_args() {
                    cx.expected_no_args(args.span().unwrap());
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
                    cx.expected_list(args.span().unwrap_or(path.span()));
                    return;
                };

                self.attribute.test_attrs.push(todo!());
            }
            _ => {
                cx.expected_specific_argument(
                    mip.span(),
                    [sym::no_crate_inject.as_str(), sym::attr.as_str()].to_vec(),
                );
            }
        }
    }

    fn add_alias<'c, S: Stage>(
        &mut self,
        cx: &'c mut AcceptContext<'_, '_, S>,
        alias: Symbol,
        span: Span,
        is_list: bool,
    ) {
        let attr_str =
            &format!("`#[doc(alias{})]`", if is_list { "(\"...\")" } else { " = \"...\"" });
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
            cx.emit_lint(AttributeLintKind::DuplicateDocAlias { first_definition }, span);
        }

        self.attribute.aliases.insert(alias, span);
    }

    fn parse_alias<'c, S: Stage>(
        &mut self,
        cx: &'c mut AcceptContext<'_, '_, S>,
        path: &PathParser<'_>,
        args: &ArgParser<'_>,
    ) {
        match args {
            ArgParser::NoArgs => {
                cx.expected_name_value(args.span().unwrap_or(path.span()), path.word_sym());
            }
            ArgParser::List(list) => {
                for i in list.mixed() {
                    let Some(alias) = i.lit().and_then(|i| i.value_str()) else {
                        cx.expected_string_literal(i.span(), i.lit());
                        continue;
                    };

                    self.add_alias(cx, alias, i.span(), false);
                }
            }
            ArgParser::NameValue(nv) => {
                let Some(alias) = nv.value_as_str() else {
                    cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
                    return;
                };
                self.add_alias(cx, alias, nv.value_span, false);
            }
        }
    }

    fn parse_keyword<'c, S: Stage>(
        &mut self,
        cx: &'c mut AcceptContext<'_, '_, S>,
        path: &PathParser<'_>,
        args: &ArgParser<'_>,
    ) {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(args.span().unwrap_or(path.span()), path.word_sym());
            return;
        };

        let Some(keyword) = nv.value_as_str() else {
            cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return;
        };

        fn is_doc_keyword(s: Symbol) -> bool {
            // FIXME: Once rustdoc can handle URL conflicts on case insensitive file systems, we
            // can remove the `SelfTy` case here, remove `sym::SelfTy`, and update the
            // `#[doc(keyword = "SelfTy")` attribute in `library/std/src/keyword_docs.rs`.
            s.is_reserved(|| edition::LATEST_STABLE_EDITION) || s.is_weak() || s == sym::SelfTy
        }

        if !is_doc_keyword(keyword) {
            cx.emit_err(DocKeywordNotKeyword { span: nv.value_span, keyword });
        }

        if self.attribute.keyword.is_some() {
            cx.duplicate_key(path.span(), path.word_sym().unwrap());
            return;
        }

        self.attribute.keyword = Some((keyword, path.span()));
    }

    fn parse_inline<'c, S: Stage>(
        &mut self,
        cx: &'c mut AcceptContext<'_, '_, S>,
        path: &PathParser<'_>,
        args: &ArgParser<'_>,
        inline: DocInline,
    ) {
        if !args.no_args() {
            cx.expected_no_args(args.span().unwrap());
            return;
        }

        let span = path.span();

        if let Some((prev_inline, prev_span)) = self.attribute.inline {
            if prev_inline == inline {
                let mut spans = MultiSpan::from_spans(vec![prev_span, span]);
                spans.push_span_label(prev_span, fluent::attr_parsing_doc_inline_conflict_first);
                spans.push_span_label(span, fluent::attr_parsing_doc_inline_conflict_second);
                cx.emit_err(DocKeywordConflict { spans });
                return;
            }
        }

        self.attribute.inline = Some((inline, span));
    }

    fn parse_single_doc_attr_item<'c, S: Stage>(
        &mut self,
        cx: &'c mut AcceptContext<'_, '_, S>,
        mip: &MetaItemParser<'_>,
    ) {
        let path = mip.path();
        let args = mip.args();

        macro_rules! no_args {
            ($ident: ident) => {{
                if !args.no_args() {
                    cx.expected_no_args(args.span().unwrap());
                    return;
                }

                if self.attribute.$ident.is_some() {
                    cx.duplicate_key(path.span(), path.word_sym().unwrap());
                    return;
                }

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

                if self.attribute.$ident.is_some() {
                    cx.duplicate_key(path.span(), path.word_sym().unwrap());
                    return;
                }

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
            Some(sym::cfg) => no_args!(cfg),
            Some(sym::cfg_hide) => no_args!(cfg_hide),
            Some(sym::notable_trait) => no_args!(notable_trait),
            Some(sym::keyword) => self.parse_keyword(cx, path, args),
            Some(sym::fake_variadic) => no_args!(fake_variadic),
            Some(sym::search_unbox) => no_args!(search_unbox),
            Some(sym::rust_logo) => no_args!(rust_logo),
            Some(sym::test) => {
                let Some(list) = args.list() else {
                    cx.expected_list(args.span().unwrap_or(path.span()));
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

                // let path = rustc_ast_pretty::pprust::path_to_string(&i_meta.path);
                // if i_meta.has_name(sym::spotlight) {
                //     self.tcx.emit_node_span_lint(
                //         INVALID_DOC_ATTRIBUTES,
                //         hir_id,
                //         i_meta.span,
                //         errors::DocTestUnknownSpotlight { path, span: i_meta.span },
                //     );
                // } else if i_meta.has_name(sym::include)
                //     && let Some(value) = i_meta.value_str()
                // {
                //     let applicability = if list.len() == 1 {
                //         Applicability::MachineApplicable
                //     } else {
                //         Applicability::MaybeIncorrect
                //     };
                //     // If there are multiple attributes, the suggestion would suggest
                //     // deleting all of them, which is incorrect.
                //     self.tcx.emit_node_span_lint(
                //         INVALID_DOC_ATTRIBUTES,
                //         hir_id,
                //         i_meta.span,
                //         errors::DocTestUnknownInclude {
                //             path,
                //             value: value.to_string(),
                //             inner: match attr.style() {
                //                 AttrStyle::Inner => "!",
                //                 AttrStyle::Outer => "",
                //             },
                //             sugg: (attr.span(), applicability),
                //         },
                //     );
                // } else if i_meta.has_name(sym::passes) || i_meta.has_name(sym::no_default_passes) {
                //     self.tcx.emit_node_span_lint(
                //         INVALID_DOC_ATTRIBUTES,
                //         hir_id,
                //         i_meta.span,
                //         errors::DocTestUnknownPasses { path, span: i_meta.span },
                //     );
                // } else if i_meta.has_name(sym::plugins) {
                //     self.tcx.emit_node_span_lint(
                //         INVALID_DOC_ATTRIBUTES,
                //         hir_id,
                //         i_meta.span,
                //         errors::DocTestUnknownPlugins { path, span: i_meta.span },
                //     );
                // } else {
                //     self.tcx.emit_node_span_lint(
                //         INVALID_DOC_ATTRIBUTES,
                //         hir_id,
                //         i_meta.span,
                //         errors::DocTestUnknownAny { path },
                //     );
                // }
            }
            _ => {
                cx.expected_specific_argument(
                    mip.span(),
                    [
                        sym::alias.as_str(),
                        sym::hidden.as_str(),
                        sym::html_favicon_url.as_str(),
                        sym::html_logo_url.as_str(),
                        sym::html_no_source.as_str(),
                        sym::html_playground_url.as_str(),
                        sym::html_root_url.as_str(),
                        sym::inline.as_str(),
                        sym::no_inline.as_str(),
                        sym::test.as_str(),
                    ]
                    .to_vec(),
                );
            }
        }
    }

    fn accept_single_doc_attr<'c, S: Stage>(
        &mut self,
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) {
        match args {
            ArgParser::NoArgs => {
                todo!()
            }
            ArgParser::List(items) => {
                for i in items.mixed() {
                    match i {
                        MetaItemOrLitParser::MetaItemParser(mip) => {
                            self.parse_single_doc_attr_item(cx, mip);
                        }
                        MetaItemOrLitParser::Lit(lit) => todo!("error should've used equals"),
                        MetaItemOrLitParser::Err(..) => {
                            // already had an error here, move on.
                        }
                    }
                }
            }
            ArgParser::NameValue(v) => {
                panic!("this should be rare if at all possible");
            }
        }
    }
}

impl<S: Stage> AttributeParser<S> for DocParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::doc],
        template!(List: "hidden|inline|...", NameValueStr: "string"),
        |this, cx, args| {
            this.accept_single_doc_attr(cx, args);
        },
    )];

    fn finalize(self, cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        todo!()
    }
}
