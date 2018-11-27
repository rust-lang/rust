use std::{
    sync::Arc,
    time::Instant,
};

use rustc_hash::FxHashMap;
use ra_syntax::{
    AstNode,  SyntaxNode, SmolStr,
    ast::{self, FnDef, FnDefNode, NameOwner, ModuleItemOwner}
};

use crate::{
    FileId, Cancelable,
    hir::{
        SourceFileItems, SourceFileItemId,
        db::HirDatabase,
        function::{FnId, FnScopes},
        module::{
            ModuleSource, ModuleSourceNode, ModuleId,
            imp::Submodule,
            nameres::{InputModuleItems, ItemMap, Resolver},
        },
    },
    input::SourceRootId,
};

/// Resolve `FnId` to the corresponding `SyntaxNode`
pub(super) fn fn_syntax(db: &impl HirDatabase, fn_id: FnId) -> FnDefNode {
    let ptr = db.id_maps().fn_ptr(fn_id);
    let syntax = db.resolve_syntax_ptr(ptr);
    FnDef::cast(syntax.borrowed()).unwrap().owned()
}

pub(super) fn fn_scopes(db: &impl HirDatabase, fn_id: FnId) -> Arc<FnScopes> {
    let syntax = db.fn_syntax(fn_id);
    let res = FnScopes::new(syntax.borrowed());
    Arc::new(res)
}

pub(super) fn file_items(db: &impl HirDatabase, file_id: FileId) -> Arc<SourceFileItems> {
    let source_file = db.file_syntax(file_id);
    let source_file = source_file.borrowed();
    let mut res = SourceFileItems::default();
    source_file
        .syntax()
        .descendants()
        .filter_map(ast::ModuleItem::cast)
        .map(|it| it.syntax().owned())
        .for_each(|it| {
            res.alloc(it);
        });
    Arc::new(res)
}

pub(super) fn file_item(
    db: &impl HirDatabase,
    file_id: FileId,
    file_item_id: SourceFileItemId,
) -> SyntaxNode {
    db.file_items(file_id)[file_item_id].clone()
}

pub(crate) fn submodules(
    db: &impl HirDatabase,
    source: ModuleSource,
) -> Cancelable<Arc<Vec<Submodule>>> {
    db.check_canceled()?;
    let file_id = source.file_id();
    let submodules = match source.resolve(db) {
        ModuleSourceNode::SourceFile(it) => collect_submodules(file_id, it.borrowed()),
        ModuleSourceNode::Module(it) => it
            .borrowed()
            .item_list()
            .map(|it| collect_submodules(file_id, it))
            .unwrap_or_else(Vec::new),
    };
    return Ok(Arc::new(submodules));

    fn collect_submodules<'a>(
        file_id: FileId,
        root: impl ast::ModuleItemOwner<'a>,
    ) -> Vec<Submodule> {
        modules(root)
            .map(|(name, m)| {
                if m.has_semi() {
                    Submodule::Declaration(name)
                } else {
                    let src = ModuleSource::new_inline(file_id, m);
                    Submodule::Definition(name, src)
                }
            })
            .collect()
    }
}

pub(crate) fn modules<'a>(
    root: impl ast::ModuleItemOwner<'a>,
) -> impl Iterator<Item = (SmolStr, ast::Module<'a>)> {
    root.items()
        .filter_map(|item| match item {
            ast::ModuleItem::Module(m) => Some(m),
            _ => None,
        })
        .filter_map(|module| {
            let name = module.name()?.text();
            Some((name, module))
        })
}

pub(super) fn input_module_items(
    db: &impl HirDatabase,
    source_root: SourceRootId,
    module_id: ModuleId,
) -> Cancelable<Arc<InputModuleItems>> {
    let module_tree = db.module_tree(source_root)?;
    let source = module_id.source(&module_tree);
    let file_items = db.file_items(source.file_id());
    let res = match source.resolve(db) {
        ModuleSourceNode::SourceFile(it) => {
            let items = it.borrowed().items();
            InputModuleItems::new(&file_items, items)
        }
        ModuleSourceNode::Module(it) => {
            let items = it
                .borrowed()
                .item_list()
                .into_iter()
                .flat_map(|it| it.items());
            InputModuleItems::new(&file_items, items)
        }
    };
    Ok(Arc::new(res))
}

pub(super) fn item_map(
    db: &impl HirDatabase,
    source_root: SourceRootId,
) -> Cancelable<Arc<ItemMap>> {
    let start = Instant::now();
    let module_tree = db.module_tree(source_root)?;
    let input = module_tree
        .modules()
        .map(|id| {
            let items = db.input_module_items(source_root, id)?;
            Ok((id, items))
        })
        .collect::<Cancelable<FxHashMap<_, _>>>()?;
    let resolver = Resolver {
        db: db,
        input: &input,
        source_root,
        module_tree,
        result: ItemMap::default(),
    };
    let res = resolver.resolve()?;
    let elapsed = start.elapsed();
    log::info!("item_map: {:?}", elapsed);
    Ok(Arc::new(res))
}
