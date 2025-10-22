//! Macro expansion utilities.

use std::mem;

use base_db::Crate;
use cfg::CfgOptions;
use drop_bomb::DropBomb;
use hir_expand::AstId;
use hir_expand::span_map::SpanMapRef;
use hir_expand::{
    ExpandError, ExpandErrorKind, ExpandResult, HirFileId, InFile, Lookup, MacroCallId,
    eager::EagerCallBackFn, mod_path::ModPath, span_map::SpanMap,
};
use span::{AstIdMap, Edition, SyntaxContext};
use syntax::ast::HasAttrs;
use syntax::{AstNode, Parse, ast};
use triomphe::Arc;
use tt::TextRange;

use crate::attr::Attrs;
use crate::expr_store::HygieneId;
use crate::macro_call_as_call_id;
use crate::nameres::DefMap;
use crate::{MacroId, UnresolvedMacro, db::DefDatabase};

#[derive(Debug)]
pub(super) struct Expander {
    span_map: SpanMap,
    current_file_id: HirFileId,
    ast_id_map: Arc<AstIdMap>,
    /// `recursion_depth == usize::MAX` indicates that the recursion limit has been reached.
    recursion_depth: u32,
    recursion_limit: usize,
}

impl Expander {
    pub(super) fn new(
        db: &dyn DefDatabase,
        current_file_id: HirFileId,
        def_map: &DefMap,
    ) -> Expander {
        let recursion_limit = def_map.recursion_limit() as usize;
        let recursion_limit = if cfg!(test) {
            // Without this, `body::tests::your_stack_belongs_to_me` stack-overflows in debug
            std::cmp::min(32, recursion_limit)
        } else {
            recursion_limit
        };
        Expander {
            current_file_id,
            recursion_depth: 0,
            recursion_limit,
            span_map: db.span_map(current_file_id),
            ast_id_map: db.ast_id_map(current_file_id),
        }
    }

    pub(super) fn ctx_for_range(&self, range: TextRange) -> SyntaxContext {
        self.span_map.span_for_range(range).ctx
    }

    pub(super) fn hygiene_for_range(&self, db: &dyn DefDatabase, range: TextRange) -> HygieneId {
        match self.span_map.as_ref() {
            hir_expand::span_map::SpanMapRef::ExpansionSpanMap(span_map) => {
                HygieneId::new(span_map.span_at(range.start()).ctx.opaque_and_semitransparent(db))
            }
            hir_expand::span_map::SpanMapRef::RealSpanMap(_) => HygieneId::ROOT,
        }
    }

    pub(super) fn is_cfg_enabled(
        &self,
        db: &dyn DefDatabase,
        has_attrs: &dyn HasAttrs,
        cfg_options: &CfgOptions,
    ) -> Result<(), cfg::CfgExpr> {
        Attrs::is_cfg_enabled_for(db, has_attrs, self.span_map.as_ref(), cfg_options)
    }

    pub(super) fn call_syntax_ctx(&self) -> SyntaxContext {
        // FIXME:
        SyntaxContext::root(Edition::CURRENT_FIXME)
    }

    pub(super) fn enter_expand<T: ast::AstNode>(
        &mut self,
        db: &dyn DefDatabase,
        macro_call: ast::MacroCall,
        krate: Crate,
        resolver: impl Fn(&ModPath) -> Option<MacroId>,
        eager_callback: EagerCallBackFn<'_>,
    ) -> Result<ExpandResult<Option<(Mark, Option<Parse<T>>)>>, UnresolvedMacro> {
        // FIXME: within_limit should support this, instead of us having to extract the error
        let mut unresolved_macro_err = None;

        let result = self.within_limit(db, |this| {
            let macro_call = this.in_file(&macro_call);

            let expands_to = hir_expand::ExpandTo::from_call_site(macro_call.value);
            let ast_id = AstId::new(macro_call.file_id, this.ast_id_map().ast_id(macro_call.value));
            let path = macro_call.value.path().and_then(|path| {
                let range = path.syntax().text_range();
                let mod_path = ModPath::from_src(db, path, &mut |range| {
                    this.span_map.span_for_range(range).ctx
                })?;
                let call_site = this.span_map.span_for_range(range);
                Some((call_site, mod_path))
            });

            let Some((call_site, path)) = path else {
                return ExpandResult::only_err(ExpandError::other(
                    this.span_map.span_for_range(macro_call.value.syntax().text_range()),
                    "malformed macro invocation",
                ));
            };

            match macro_call_as_call_id(
                db,
                ast_id,
                &path,
                call_site.ctx,
                expands_to,
                krate,
                |path| resolver(path).map(|it| db.macro_def(it)),
                eager_callback,
            ) {
                Ok(call_id) => call_id,
                Err(resolve_err) => {
                    unresolved_macro_err = Some(resolve_err);
                    ExpandResult { value: None, err: None }
                }
            }
        });

        if let Some(err) = unresolved_macro_err { Err(err) } else { Ok(result) }
    }

    pub(super) fn enter_expand_id<T: ast::AstNode>(
        &mut self,
        db: &dyn DefDatabase,
        call_id: MacroCallId,
    ) -> ExpandResult<Option<(Mark, Option<Parse<T>>)>> {
        self.within_limit(db, |_this| ExpandResult::ok(Some(call_id)))
    }

    pub(super) fn exit(&mut self, Mark { file_id, span_map, ast_id_map, mut bomb }: Mark) {
        self.span_map = span_map;
        self.current_file_id = file_id;
        self.ast_id_map = ast_id_map;
        if self.recursion_depth == u32::MAX {
            // Recursion limit has been reached somewhere in the macro expansion tree. Reset the
            // depth only when we get out of the tree.
            if !self.current_file_id.is_macro() {
                self.recursion_depth = 0;
            }
        } else {
            self.recursion_depth -= 1;
        }
        bomb.defuse();
    }

    pub(super) fn in_file<T>(&self, value: T) -> InFile<T> {
        InFile { file_id: self.current_file_id, value }
    }

    pub(super) fn current_file_id(&self) -> HirFileId {
        self.current_file_id
    }

    fn within_limit<F, T: ast::AstNode>(
        &mut self,
        db: &dyn DefDatabase,
        op: F,
    ) -> ExpandResult<Option<(Mark, Option<Parse<T>>)>>
    where
        F: FnOnce(&mut Self) -> ExpandResult<Option<MacroCallId>>,
    {
        if self.recursion_depth == u32::MAX {
            // Recursion limit has been reached somewhere in the macro expansion tree. We should
            // stop expanding other macro calls in this tree, or else this may result in
            // exponential number of macro expansions, leading to a hang.
            //
            // The overflow error should have been reported when it occurred (see the next branch),
            // so don't return overflow error here to avoid diagnostics duplication.
            cov_mark::hit!(overflow_but_not_me);
            return ExpandResult::ok(None);
        }

        let ExpandResult { value, err } = op(self);
        let Some(call_id) = value else {
            return ExpandResult { value: None, err };
        };
        if self.recursion_depth as usize > self.recursion_limit {
            self.recursion_depth = u32::MAX;
            cov_mark::hit!(your_stack_belongs_to_me);
            return ExpandResult::only_err(ExpandError::new(
                db.macro_arg_considering_derives(call_id, &call_id.lookup(db).kind).2,
                ExpandErrorKind::RecursionOverflow,
            ));
        }

        let res = db.parse_macro_expansion(call_id);

        let err = err.or(res.err);
        ExpandResult {
            value: {
                let parse = res.value.0.cast::<T>();

                self.recursion_depth += 1;
                let old_file_id = std::mem::replace(&mut self.current_file_id, call_id.into());
                let old_span_map =
                    std::mem::replace(&mut self.span_map, db.span_map(self.current_file_id));
                let prev_ast_id_map =
                    mem::replace(&mut self.ast_id_map, db.ast_id_map(self.current_file_id));
                let mark = Mark {
                    file_id: old_file_id,
                    span_map: old_span_map,
                    ast_id_map: prev_ast_id_map,
                    bomb: DropBomb::new("expansion mark dropped"),
                };
                Some((mark, parse))
            },
            err,
        }
    }

    #[inline]
    pub(super) fn ast_id_map(&self) -> &AstIdMap {
        &self.ast_id_map
    }

    #[inline]
    pub(super) fn span_map(&self) -> SpanMapRef<'_> {
        self.span_map.as_ref()
    }
}

#[derive(Debug)]
pub(super) struct Mark {
    file_id: HirFileId,
    span_map: SpanMap,
    ast_id_map: Arc<AstIdMap>,
    bomb: DropBomb,
}
