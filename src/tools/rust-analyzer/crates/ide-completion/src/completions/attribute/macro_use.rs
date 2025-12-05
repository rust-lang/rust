//! Completion for macros in `#[macro_use(...)]`
use hir::ModuleDef;
use ide_db::SymbolKind;
use syntax::ast;

use crate::{Completions, context::CompletionContext, item::CompletionItem};

pub(super) fn complete_macro_use(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    extern_crate: Option<&ast::ExternCrate>,
    existing_imports: &[ast::Path],
) {
    let Some(extern_crate) = extern_crate else { return };
    let Some(extern_crate) = ctx.sema.to_def(extern_crate) else { return };
    let Some(krate) = extern_crate.resolved_crate(ctx.db) else { return };

    for mod_def in krate.root_module().declarations(ctx.db) {
        if let ModuleDef::Macro(mac) = mod_def {
            let mac_name = mac.name(ctx.db);
            let mac_name = mac_name.as_str();

            let existing_import = existing_imports
                .iter()
                .filter_map(|p| p.as_single_name_ref())
                .find(|n| n.text() == mac_name);
            if existing_import.is_some() {
                continue;
            }

            let item =
                CompletionItem::new(SymbolKind::Macro, ctx.source_range(), mac_name, ctx.edition);
            item.add_to(acc, ctx.db);
        }
    }
}
