//! Macro expansion utilities.

use std::cell::OnceCell;

use base_db::CrateId;
use cfg::CfgOptions;
use drop_bomb::DropBomb;
use hir_expand::{
    attrs::RawAttrs, mod_path::ModPath, span_map::SpanMap, ExpandError, ExpandErrorKind,
    ExpandResult, HirFileId, InFile, Lookup, MacroCallId,
};
use span::{Edition, SyntaxContextId};
use syntax::{ast, Parse};
use triomphe::Arc;

use crate::type_ref::{TypesMap, TypesSourceMap};
use crate::{
    attr::Attrs, db::DefDatabase, lower::LowerCtx, path::Path, AsMacroCall, MacroId, ModuleId,
    UnresolvedMacro,
};

#[derive(Debug)]
pub struct Expander {
    cfg_options: Arc<CfgOptions>,
    span_map: OnceCell<SpanMap>,
    current_file_id: HirFileId,
    pub(crate) module: ModuleId,
    /// `recursion_depth == usize::MAX` indicates that the recursion limit has been reached.
    recursion_depth: u32,
    recursion_limit: usize,
}

impl Expander {
    pub fn new(db: &dyn DefDatabase, current_file_id: HirFileId, module: ModuleId) -> Expander {
        let recursion_limit = module.def_map(db).recursion_limit() as usize;
        let recursion_limit = if cfg!(test) {
            // Without this, `body::tests::your_stack_belongs_to_me` stack-overflows in debug
            std::cmp::min(32, recursion_limit)
        } else {
            recursion_limit
        };
        Expander {
            current_file_id,
            module,
            recursion_depth: 0,
            recursion_limit,
            cfg_options: db.crate_graph()[module.krate].cfg_options.clone(),
            span_map: OnceCell::new(),
        }
    }

    pub(crate) fn span_map(&self, db: &dyn DefDatabase) -> &SpanMap {
        self.span_map.get_or_init(|| db.span_map(self.current_file_id))
    }

    pub fn krate(&self) -> CrateId {
        self.module.krate
    }

    pub fn syntax_context(&self) -> SyntaxContextId {
        // FIXME:
        SyntaxContextId::root(Edition::CURRENT)
    }

    pub fn enter_expand<T: ast::AstNode>(
        &mut self,
        db: &dyn DefDatabase,
        macro_call: ast::MacroCall,
        resolver: impl Fn(&ModPath) -> Option<MacroId>,
    ) -> Result<ExpandResult<Option<(Mark, Parse<T>)>>, UnresolvedMacro> {
        // FIXME: within_limit should support this, instead of us having to extract the error
        let mut unresolved_macro_err = None;

        let result = self.within_limit(db, |this| {
            let macro_call = this.in_file(&macro_call);
            match macro_call.as_call_id_with_errors(db.upcast(), this.module.krate(), |path| {
                resolver(path).map(|it| db.macro_def(it))
            }) {
                Ok(call_id) => call_id,
                Err(resolve_err) => {
                    unresolved_macro_err = Some(resolve_err);
                    ExpandResult { value: None, err: None }
                }
            }
        });

        if let Some(err) = unresolved_macro_err {
            Err(err)
        } else {
            Ok(result)
        }
    }

    pub fn enter_expand_id<T: ast::AstNode>(
        &mut self,
        db: &dyn DefDatabase,
        call_id: MacroCallId,
    ) -> ExpandResult<Option<(Mark, Parse<T>)>> {
        self.within_limit(db, |_this| ExpandResult::ok(Some(call_id)))
    }

    pub fn exit(&mut self, mut mark: Mark) {
        self.span_map = mark.span_map;
        self.current_file_id = mark.file_id;
        if self.recursion_depth == u32::MAX {
            // Recursion limit has been reached somewhere in the macro expansion tree. Reset the
            // depth only when we get out of the tree.
            if !self.current_file_id.is_macro() {
                self.recursion_depth = 0;
            }
        } else {
            self.recursion_depth -= 1;
        }
        mark.bomb.defuse();
    }

    pub fn ctx<'a>(
        &self,
        db: &'a dyn DefDatabase,
        types_map: &'a mut TypesMap,
        types_source_map: &'a mut TypesSourceMap,
    ) -> LowerCtx<'a> {
        LowerCtx::with_span_map_cell(
            db,
            self.current_file_id,
            self.span_map.clone(),
            types_map,
            types_source_map,
        )
    }

    pub(crate) fn in_file<T>(&self, value: T) -> InFile<T> {
        InFile { file_id: self.current_file_id, value }
    }

    pub(crate) fn parse_attrs(&self, db: &dyn DefDatabase, owner: &dyn ast::HasAttrs) -> Attrs {
        Attrs::filter(
            db,
            self.krate(),
            RawAttrs::new(
                db.upcast(),
                owner,
                self.span_map.get_or_init(|| db.span_map(self.current_file_id)).as_ref(),
            ),
        )
    }

    pub(crate) fn cfg_options(&self) -> &CfgOptions {
        &self.cfg_options
    }

    pub fn current_file_id(&self) -> HirFileId {
        self.current_file_id
    }

    pub(crate) fn parse_path(
        &mut self,
        db: &dyn DefDatabase,
        path: ast::Path,
        types_map: &mut TypesMap,
        types_source_map: &mut TypesSourceMap,
    ) -> Option<Path> {
        let mut ctx = LowerCtx::with_span_map_cell(
            db,
            self.current_file_id,
            self.span_map.clone(),
            types_map,
            types_source_map,
        );
        Path::from_src(&mut ctx, path)
    }

    fn within_limit<F, T: ast::AstNode>(
        &mut self,
        db: &dyn DefDatabase,
        op: F,
    ) -> ExpandResult<Option<(Mark, Parse<T>)>>
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
                db.macro_arg_considering_derives(call_id, &call_id.lookup(db.upcast()).kind).2,
                ExpandErrorKind::RecursionOverflow,
            ));
        }

        let macro_file = call_id.as_macro_file();
        let res = db.parse_macro_expansion(macro_file);

        let err = err.or(res.err);
        ExpandResult {
            value: match &err {
                // If proc-macro is disabled or unresolved, we want to expand to a missing expression
                // instead of an empty tree which might end up in an empty block.
                Some(e) if matches!(e.kind(), ExpandErrorKind::MissingProcMacroExpander(_)) => None,
                _ => (|| {
                    let parse = res.value.0.cast::<T>()?;

                    self.recursion_depth += 1;
                    let old_span_map = OnceCell::new();
                    if let Some(prev) = self.span_map.take() {
                        _ = old_span_map.set(prev);
                    };
                    _ = self.span_map.set(SpanMap::ExpansionSpanMap(res.value.1));
                    let old_file_id =
                        std::mem::replace(&mut self.current_file_id, macro_file.into());
                    let mark = Mark {
                        file_id: old_file_id,
                        span_map: old_span_map,
                        bomb: DropBomb::new("expansion mark dropped"),
                    };
                    Some((mark, parse))
                })(),
            },
            err,
        }
    }
}

#[derive(Debug)]
pub struct Mark {
    file_id: HirFileId,
    span_map: OnceCell<SpanMap>,
    bomb: DropBomb,
}
