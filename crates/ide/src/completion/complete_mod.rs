//! Completes mod declarations.

use base_db::FileLoader;
use hir::ModuleSource;

use super::{completion_context::CompletionContext, completion_item::Completions};

/// Complete mod declaration, i.e. `mod <|> ;`
pub(super) fn complete_mod(acc: &mut Completions, ctx: &CompletionContext) {
    let module_names_for_import = ctx
        .sema
        // TODO kb this is wrong, since we need not the file module
        .to_module_def(ctx.position.file_id)
        .and_then(|current_module| {
            dbg!(current_module.name(ctx.db));
            dbg!(current_module.definition_source(ctx.db));
            dbg!(current_module.declaration_source(ctx.db));
            let mut zz = Vec::new();
            let mut vv = Some(current_module);
            while let Some(ModuleSource::Module(_)) =
                vv.map(|vv| vv.definition_source(ctx.db).value)
            {
                zz.push(current_module.name(ctx.db));
                vv = current_module.parent(ctx.db);
            }
            dbg!(zz);
            let definition_source = current_module.definition_source(ctx.db);
            // TODO kb filter out declarations in possible_sudmobule_names
            // let declaration_source = current_module.declaration_source(ctx.db);
            let module_definition_source_file = definition_source.file_id.original_file(ctx.db);
            let mod_declaration_candidates =
                ctx.db.possible_sudmobule_names(module_definition_source_file);
            dbg!(mod_declaration_candidates);
            // TODO kb exlude existing children from the candidates
            let existing_children = current_module.children(ctx.db).collect::<Vec<_>>();
            None::<Vec<String>>
        })
        .unwrap_or_default();
}
