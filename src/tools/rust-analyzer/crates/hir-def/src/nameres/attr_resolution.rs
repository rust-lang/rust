//! Post-nameres attribute resolution.

use base_db::Crate;
use hir_expand::{
    MacroCallId, MacroCallKind, MacroDefId,
    attrs::{Attr, AttrId, AttrInput},
    inert_attr_macro::find_builtin_attr_idx,
    mod_path::{ModPath, PathKind},
};
use span::SyntaxContext;
use syntax::ast;
use triomphe::Arc;

use crate::{
    AstIdWithPath, LocalModuleId, MacroId, UnresolvedMacro,
    db::DefDatabase,
    item_scope::BuiltinShadowMode,
    nameres::{LocalDefMap, path_resolution::ResolveMode},
};

use super::{DefMap, MacroSubNs};

pub enum ResolvedAttr {
    /// Attribute resolved to an attribute macro.
    Macro(MacroCallId),
    /// Attribute resolved to something else that does not require expansion.
    Other,
}

impl DefMap {
    pub(crate) fn resolve_attr_macro(
        &self,
        local_def_map: &LocalDefMap,
        db: &dyn DefDatabase,
        original_module: LocalModuleId,
        ast_id: AstIdWithPath<ast::Item>,
        attr: &Attr,
    ) -> Result<ResolvedAttr, UnresolvedMacro> {
        // NB: does not currently work for derive helpers as they aren't recorded in the `DefMap`

        if self.is_builtin_or_registered_attr(&ast_id.path) {
            return Ok(ResolvedAttr::Other);
        }

        let resolved_res = self.resolve_path_fp_with_macro(
            local_def_map,
            db,
            ResolveMode::Other,
            original_module,
            &ast_id.path,
            BuiltinShadowMode::Module,
            Some(MacroSubNs::Attr),
        );
        let def = match resolved_res.resolved_def.take_macros() {
            Some(def) => {
                // `MacroSubNs` is just a hint, so the path may still resolve to a custom derive
                // macro, or even function-like macro when the path is qualified.
                if def.is_attribute(db) {
                    def
                } else {
                    return Ok(ResolvedAttr::Other);
                }
            }
            None => return Err(UnresolvedMacro { path: ast_id.path.as_ref().clone() }),
        };

        Ok(ResolvedAttr::Macro(attr_macro_as_call_id(
            db,
            &ast_id,
            attr,
            self.krate,
            db.macro_def(def),
        )))
    }

    pub(crate) fn is_builtin_or_registered_attr(&self, path: &ModPath) -> bool {
        if path.kind != PathKind::Plain {
            return false;
        }

        let segments = path.segments();

        if let Some(name) = segments.first() {
            let name = name.symbol();
            let pred = |n: &_| *n == *name;

            let is_tool = self.data.registered_tools.iter().any(pred);
            // FIXME: tool modules can be shadowed by actual modules
            if is_tool {
                return true;
            }

            if segments.len() == 1 {
                if find_builtin_attr_idx(name).is_some() {
                    return true;
                }
                if self.data.registered_attrs.iter().any(pred) {
                    return true;
                }
            }
        }
        false
    }
}

pub(super) fn attr_macro_as_call_id(
    db: &dyn DefDatabase,
    item_attr: &AstIdWithPath<ast::Item>,
    macro_attr: &Attr,
    krate: Crate,
    def: MacroDefId,
) -> MacroCallId {
    let arg = match macro_attr.input.as_deref() {
        Some(AttrInput::TokenTree(tt)) => {
            let mut tt = tt.clone();
            tt.top_subtree_delimiter_mut().kind = tt::DelimiterKind::Invisible;
            Some(tt)
        }

        _ => None,
    };

    def.make_call(
        db,
        krate,
        MacroCallKind::Attr {
            ast_id: item_attr.ast_id,
            attr_args: arg.map(Arc::new),
            invoc_attr_index: macro_attr.id,
        },
        macro_attr.ctxt,
    )
}

pub(super) fn derive_macro_as_call_id(
    db: &dyn DefDatabase,
    item_attr: &AstIdWithPath<ast::Adt>,
    derive_attr_index: AttrId,
    derive_pos: u32,
    call_site: SyntaxContext,
    krate: Crate,
    resolver: impl Fn(&ModPath) -> Option<(MacroId, MacroDefId)>,
    derive_macro_id: MacroCallId,
) -> Result<(MacroId, MacroDefId, MacroCallId), UnresolvedMacro> {
    let (macro_id, def_id) = resolver(&item_attr.path)
        .filter(|(_, def_id)| def_id.is_derive())
        .ok_or_else(|| UnresolvedMacro { path: item_attr.path.as_ref().clone() })?;
    let call_id = def_id.make_call(
        db,
        krate,
        MacroCallKind::Derive {
            ast_id: item_attr.ast_id,
            derive_index: derive_pos,
            derive_attr_index,
            derive_macro_id,
        },
        call_site,
    );
    Ok((macro_id, def_id, call_id))
}
