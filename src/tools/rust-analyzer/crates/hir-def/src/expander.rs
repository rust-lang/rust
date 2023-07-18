//! Macro expansion utilities.

use base_db::CrateId;
use cfg::CfgOptions;
use drop_bomb::DropBomb;
use hir_expand::{
    attrs::RawAttrs, hygiene::Hygiene, mod_path::ModPath, ExpandError, ExpandResult, HirFileId,
    InFile, MacroCallId, UnresolvedMacro,
};
use limit::Limit;
use syntax::{ast, Parse, SyntaxNode};

use crate::{
    attr::Attrs, db::DefDatabase, lower::LowerCtx, macro_id_to_def_id, path::Path, AsMacroCall,
    MacroId, ModuleId,
};

#[derive(Debug)]
pub struct Expander {
    cfg_options: CfgOptions,
    hygiene: Hygiene,
    krate: CrateId,
    pub(crate) current_file_id: HirFileId,
    pub(crate) module: ModuleId,
    /// `recursion_depth == usize::MAX` indicates that the recursion limit has been reached.
    recursion_depth: u32,
    recursion_limit: Limit,
}

impl Expander {
    pub fn new(db: &dyn DefDatabase, current_file_id: HirFileId, module: ModuleId) -> Expander {
        let recursion_limit = db.recursion_limit(module.krate);
        #[cfg(not(test))]
        let recursion_limit = Limit::new(recursion_limit as usize);
        // Without this, `body::tests::your_stack_belongs_to_me` stack-overflows in debug
        #[cfg(test)]
        let recursion_limit = Limit::new(std::cmp::min(32, recursion_limit as usize));
        Expander {
            current_file_id,
            module,
            recursion_depth: 0,
            recursion_limit,
            cfg_options: db.crate_graph()[module.krate].cfg_options.clone(),
            hygiene: Hygiene::new(db.upcast(), current_file_id),
            krate: module.krate,
        }
    }

    pub fn enter_expand<T: ast::AstNode>(
        &mut self,
        db: &dyn DefDatabase,
        macro_call: ast::MacroCall,
        resolver: impl Fn(ModPath) -> Option<MacroId>,
    ) -> Result<ExpandResult<Option<(Mark, Parse<T>)>>, UnresolvedMacro> {
        // FIXME: within_limit should support this, instead of us having to extract the error
        let mut unresolved_macro_err = None;

        let result = self.within_limit(db, |this| {
            let macro_call = InFile::new(this.current_file_id, &macro_call);
            match macro_call.as_call_id_with_errors(db.upcast(), this.module.krate(), |path| {
                resolver(path).map(|it| macro_id_to_def_id(db, it))
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

    fn enter_expand_inner(
        db: &dyn DefDatabase,
        call_id: MacroCallId,
        error: Option<ExpandError>,
    ) -> ExpandResult<Option<InFile<Parse<SyntaxNode>>>> {
        let macro_file = call_id.as_macro_file();
        let ExpandResult { value, err } = db.parse_macro_expansion(macro_file);

        ExpandResult { value: Some(InFile::new(macro_file.into(), value.0)), err: error.or(err) }
    }

    pub fn exit(&mut self, db: &dyn DefDatabase, mut mark: Mark) {
        self.hygiene = Hygiene::new(db.upcast(), mark.file_id);
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

    pub fn ctx<'a>(&self, db: &'a dyn DefDatabase) -> LowerCtx<'a> {
        LowerCtx::new(db, &self.hygiene, self.current_file_id)
    }

    pub(crate) fn to_source<T>(&self, value: T) -> InFile<T> {
        InFile { file_id: self.current_file_id, value }
    }

    pub(crate) fn parse_attrs(&self, db: &dyn DefDatabase, owner: &dyn ast::HasAttrs) -> Attrs {
        Attrs::filter(db, self.krate, RawAttrs::new(db.upcast(), owner, &self.hygiene))
    }

    pub(crate) fn cfg_options(&self) -> &CfgOptions {
        &self.cfg_options
    }

    pub fn current_file_id(&self) -> HirFileId {
        self.current_file_id
    }

    pub(crate) fn parse_path(&mut self, db: &dyn DefDatabase, path: ast::Path) -> Option<Path> {
        let ctx = LowerCtx::new(db, &self.hygiene, self.current_file_id);
        Path::from_src(path, &ctx)
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
            return ExpandResult::only_err(ExpandError::RecursionOverflowPoisoned);
        } else if self.recursion_limit.check(self.recursion_depth as usize + 1).is_err() {
            self.recursion_depth = u32::MAX;
            cov_mark::hit!(your_stack_belongs_to_me);
            return ExpandResult::only_err(ExpandError::other(
                "reached recursion limit during macro expansion",
            ));
        }

        let ExpandResult { value, err } = op(self);
        let Some(call_id) = value else {
            return ExpandResult { value: None, err };
        };

        Self::enter_expand_inner(db, call_id, err).map(|value| {
            value.and_then(|InFile { file_id, value }| {
                let parse = value.cast::<T>()?;

                self.recursion_depth += 1;
                self.hygiene = Hygiene::new(db.upcast(), file_id);
                let old_file_id = std::mem::replace(&mut self.current_file_id, file_id);
                let mark =
                    Mark { file_id: old_file_id, bomb: DropBomb::new("expansion mark dropped") };
                Some((mark, parse))
            })
        })
    }
}

#[derive(Debug)]
pub struct Mark {
    file_id: HirFileId,
    bomb: DropBomb,
}
