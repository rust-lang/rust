//! Defines attribute helpers for name resolution.
//!
//! Notice we don't preserve all attributes for name resolution, to save space:
//! for example, we skip doc comments (desugared to `#[doc = "..."]` attributes)
//! and `#[inline]`. The filtered attributes are listed in [`hir_expand::attrs`].

use std::{
    borrow::Cow,
    convert::Infallible,
    ops::{self, ControlFlow},
};

use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{
    attrs::{Attr, AttrId, AttrInput, Meta, collect_item_tree_attrs},
    mod_path::ModPath,
    name::Name,
    span_map::SpanMapRef,
};
use intern::{Interned, Symbol, sym};
use syntax::{AstNode, T, ast};
use syntax_bridge::DocCommentDesugarMode;
use tt::token_to_literal;

use crate::{db::DefDatabase, item_tree::lower::Ctx};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AttrsOrCfg {
    Enabled {
        attrs: AttrsOwned,
    },
    /// This only collects the attributes up to the disabled `cfg` (this is what needed for crate-level attributes.)
    CfgDisabled(Box<(CfgExpr, AttrsOwned)>),
}

impl Default for AttrsOrCfg {
    #[inline]
    fn default() -> Self {
        AttrsOrCfg::Enabled { attrs: AttrsOwned(Box::new([])) }
    }
}

impl AttrsOrCfg {
    pub(crate) fn lower<'a>(
        db: &dyn DefDatabase,
        owner: &dyn ast::HasAttrs,
        cfg_options: &dyn Fn() -> &'a CfgOptions,
        span_map: SpanMapRef<'_>,
    ) -> AttrsOrCfg {
        let mut attrs = Vec::new();
        let result =
            collect_item_tree_attrs::<Infallible>(owner, cfg_options, |meta, container, _, _| {
                // NOTE: We cannot early return from this function, *every* attribute must be pushed, otherwise we'll mess the `AttrId`
                // tracking.
                let (span, path_range, input) = match meta {
                    Meta::NamedKeyValue { path_range, name: _, value } => {
                        let span = span_map.span_for_range(path_range);
                        let input = value.map(|value| {
                            Box::new(AttrInput::Literal(token_to_literal(
                                value.text(),
                                span_map.span_for_range(value.text_range()),
                            )))
                        });
                        (span, path_range, input)
                    }
                    Meta::TokenTree { path, tt } => {
                        let span = span_map.span_for_range(path.range);
                        let tt = syntax_bridge::syntax_node_to_token_tree(
                            tt.syntax(),
                            span_map,
                            span,
                            DocCommentDesugarMode::ProcMacro,
                        );
                        let input = Some(Box::new(AttrInput::TokenTree(tt)));
                        (span, path.range, input)
                    }
                    Meta::Path { path } => {
                        let span = span_map.span_for_range(path.range);
                        (span, path.range, None)
                    }
                };

                let path = container.token_at_offset(path_range.start()).right_biased().and_then(
                    |first_path_token| {
                        let is_abs = matches!(first_path_token.kind(), T![:] | T![::]);
                        let segments =
                            std::iter::successors(Some(first_path_token), |it| it.next_token())
                                .take_while(|it| it.text_range().end() <= path_range.end())
                                .filter(|it| it.kind().is_any_identifier());
                        ModPath::from_tokens(
                            db,
                            &mut |range| span_map.span_for_range(range).ctx,
                            is_abs,
                            segments,
                        )
                    },
                );
                let path = path.unwrap_or_else(|| Name::missing().into());

                attrs.push(Attr { path: Interned::new(path), input, ctxt: span.ctx });
                ControlFlow::Continue(())
            });
        let attrs = AttrsOwned(attrs.into_boxed_slice());
        match result {
            Some(Either::Right(cfg)) => AttrsOrCfg::CfgDisabled(Box::new((cfg, attrs))),
            None => AttrsOrCfg::Enabled { attrs },
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AttrsOwned(Box<[Attr]>);

#[derive(Debug, Clone, Copy)]
pub(crate) struct Attrs<'a>(&'a [Attr]);

impl ops::Deref for Attrs<'_> {
    type Target = [Attr];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl Ctx<'_> {
    #[inline]
    pub(super) fn lower_attrs(&self, owner: &dyn ast::HasAttrs) -> AttrsOrCfg {
        AttrsOrCfg::lower(self.db, owner, &|| self.cfg_options(), self.span_map())
    }
}

impl AttrsOwned {
    #[inline]
    pub(crate) fn as_ref(&self) -> Attrs<'_> {
        Attrs(&self.0)
    }
}

impl<'a> Attrs<'a> {
    pub(crate) const EMPTY: Self = Attrs(&[]);

    #[inline]
    pub(crate) fn by_key(self, key: Symbol) -> AttrQuery<'a> {
        AttrQuery { attrs: self, key }
    }

    #[inline]
    pub(crate) fn iter(self) -> impl Iterator<Item = (AttrId, &'a Attr)> {
        self.0.iter().enumerate().map(|(id, attr)| (AttrId::from_item_tree_index(id as u32), attr))
    }

    #[inline]
    pub(crate) fn iter_after(
        self,
        after: Option<AttrId>,
    ) -> impl Iterator<Item = (AttrId, &'a Attr)> {
        let skip = after.map_or(0, |after| after.item_tree_index() + 1);
        self.0[skip as usize..]
            .iter()
            .enumerate()
            .map(move |(id, attr)| (AttrId::from_item_tree_index(id as u32 + skip), attr))
    }

    #[inline]
    pub(crate) fn is_proc_macro(&self) -> bool {
        self.by_key(sym::proc_macro).exists()
    }

    #[inline]
    pub(crate) fn is_proc_macro_attribute(&self) -> bool {
        self.by_key(sym::proc_macro_attribute).exists()
    }
}
#[derive(Debug, Clone)]
pub(crate) struct AttrQuery<'attr> {
    attrs: Attrs<'attr>,
    key: Symbol,
}

impl<'attr> AttrQuery<'attr> {
    #[inline]
    pub(crate) fn tt_values(self) -> impl Iterator<Item = &'attr crate::tt::TopSubtree> {
        self.attrs().filter_map(|attr| attr.token_tree_value())
    }

    #[inline]
    pub(crate) fn string_value_with_span(self) -> Option<(&'attr Symbol, span::Span)> {
        self.attrs().find_map(|attr| attr.string_value_with_span())
    }

    #[inline]
    pub(crate) fn string_value_unescape(self) -> Option<Cow<'attr, str>> {
        self.attrs().find_map(|attr| attr.string_value_unescape())
    }

    #[inline]
    pub(crate) fn exists(self) -> bool {
        self.attrs().next().is_some()
    }

    #[inline]
    pub(crate) fn attrs(self) -> impl Iterator<Item = &'attr Attr> + Clone {
        let key = self.key;
        self.attrs.0.iter().filter(move |attr| attr.path.as_ident().is_some_and(|s| *s == key))
    }
}

impl AttrsOrCfg {
    #[inline]
    pub(super) fn empty() -> Self {
        AttrsOrCfg::Enabled { attrs: AttrsOwned(Box::new([])) }
    }

    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        matches!(self, AttrsOrCfg::Enabled { attrs } if attrs.as_ref().is_empty())
    }
}
