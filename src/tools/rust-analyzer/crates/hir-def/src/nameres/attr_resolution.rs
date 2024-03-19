//! Post-nameres attribute resolution.

use base_db::CrateId;
use hir_expand::{
    attrs::{Attr, AttrId, AttrInput},
    MacroCallId, MacroCallKind, MacroDefId,
};
use span::SyntaxContextId;
use syntax::{ast, SmolStr};
use triomphe::Arc;

use crate::{
    attr::builtin::{find_builtin_attr_idx, TOOL_MODULES},
    db::DefDatabase,
    item_scope::BuiltinShadowMode,
    nameres::path_resolution::ResolveMode,
    path::{self, ModPath, PathKind},
    AstIdWithPath, LocalModuleId, MacroId, UnresolvedMacro,
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
            db.macro_def(def),
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

pub(super) fn attr_macro_as_call_id(
    db: &dyn DefDatabase,
    item_attr: &AstIdWithPath<ast::Item>,
    macro_attr: &Attr,
    krate: CrateId,
    def: MacroDefId,
) -> MacroCallId {
    let arg = match macro_attr.input.as_deref() {
        Some(AttrInput::TokenTree(tt)) => {
            let mut tt = tt.as_ref().clone();
            tt.delimiter.kind = tt::DelimiterKind::Invisible;
            Some(tt)
        }

        _ => None,
    };

    def.make_call(
        db.upcast(),
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
    call_site: SyntaxContextId,
    krate: CrateId,
    resolver: impl Fn(path::ModPath) -> Option<(MacroId, MacroDefId)>,
) -> Result<(MacroId, MacroDefId, MacroCallId), UnresolvedMacro> {
    let (macro_id, def_id) = resolver(item_attr.path.clone())
        .filter(|(_, def_id)| def_id.is_derive())
        .ok_or_else(|| UnresolvedMacro { path: item_attr.path.clone() })?;
    let call_id = def_id.make_call(
        db.upcast(),
        krate,
        MacroCallKind::Derive {
            ast_id: item_attr.ast_id,
            derive_index: derive_pos,
            derive_attr_index,
        },
        call_site,
    );
    Ok((macro_id, def_id, call_id))
}
