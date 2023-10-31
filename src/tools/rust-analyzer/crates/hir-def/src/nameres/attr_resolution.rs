//! Post-nameres attribute resolution.

use hir_expand::{attrs::Attr, MacroCallId};
use syntax::{ast, SmolStr};

use crate::{
    attr::builtin::{find_builtin_attr_idx, TOOL_MODULES},
    attr_macro_as_call_id,
    db::DefDatabase,
    item_scope::BuiltinShadowMode,
    macro_id_to_def_id,
    nameres::path_resolution::ResolveMode,
    path::{ModPath, PathKind},
    AstIdWithPath, LocalModuleId, UnresolvedMacro,
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
            None => return Err(UnresolvedMacro { path: ast_id.path }),
        };

        Ok(ResolvedAttr::Macro(attr_macro_as_call_id(
            db,
            &ast_id,
            attr,
            self.krate,
            macro_id_to_def_id(db, def),
        )))
    }

    pub(crate) fn is_builtin_or_registered_attr(&self, path: &ModPath) -> bool {
        if path.kind != PathKind::Plain {
            return false;
        }

        let segments = path.segments();

        if let Some(name) = segments.first() {
            let name = name.to_smol_str();
            let pred = |n: &_| *n == name;

            let registered = self.data.registered_tools.iter().map(SmolStr::as_str);
            let is_tool = TOOL_MODULES.iter().copied().chain(registered).any(pred);
            // FIXME: tool modules can be shadowed by actual modules
            if is_tool {
                return true;
            }

            if segments.len() == 1 {
                let mut registered = self.data.registered_attrs.iter().map(SmolStr::as_str);
                let is_inert = find_builtin_attr_idx(&name).is_some() || registered.any(pred);
                return is_inert;
            }
        }
        false
    }
}
