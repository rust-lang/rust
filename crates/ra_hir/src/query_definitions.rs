use std::{
    sync::Arc,
    time::Instant,
};

use rustc_hash::FxHashMap;
use ra_syntax::{
    AstNode, SyntaxNode,
    ast::{self, NameOwner, ModuleItemOwner}
};
use ra_db::{SourceRootId, Cancelable,};

use crate::{
    SourceFileItems, SourceItemId, DefKind, Function, DefId, Name, AsName, HirFileId,
    MacroCallLoc,
    db::HirDatabase,
    function::FnScopes,
    module::{
        ModuleSource, ModuleSourceNode, ModuleId,
        imp::Submodule,
        nameres::{InputModuleItems, ItemMap, Resolver},
    },
    adt::{StructData, EnumData},
};

pub(super) fn fn_scopes(db: &impl HirDatabase, def_id: DefId) -> Arc<FnScopes> {
    let function = Function::new(def_id);
    let syntax = function.syntax(db);
    let res = FnScopes::new(syntax.borrowed());
    Arc::new(res)
}

pub(super) fn struct_data(db: &impl HirDatabase, def_id: DefId) -> Cancelable<Arc<StructData>> {
    let def_loc = def_id.loc(db);
    assert!(def_loc.kind == DefKind::Struct);
    let syntax = db.file_item(def_loc.source_item_id);
    let struct_def =
        ast::StructDef::cast(syntax.borrowed()).expect("struct def should point to StructDef node");
    Ok(Arc::new(StructData::new(struct_def.borrowed())))
}

pub(super) fn enum_data(db: &impl HirDatabase, def_id: DefId) -> Cancelable<Arc<EnumData>> {
    let def_loc = def_id.loc(db);
    assert!(def_loc.kind == DefKind::Enum);
    let syntax = db.file_item(def_loc.source_item_id);
    let enum_def =
        ast::EnumDef::cast(syntax.borrowed()).expect("enum def should point to EnumDef node");
    Ok(Arc::new(EnumData::new(enum_def.borrowed())))
}

pub(super) fn file_items(db: &impl HirDatabase, file_id: HirFileId) -> Arc<SourceFileItems> {
    let source_file = db.hir_source_file(file_id);
    let source_file = source_file.borrowed();
    let res = SourceFileItems::new(file_id, source_file);
    Arc::new(res)
}

pub(super) fn file_item(db: &impl HirDatabase, source_item_id: SourceItemId) -> SyntaxNode {
    match source_item_id.item_id {
        Some(id) => db.file_items(source_item_id.file_id)[id].clone(),
        None => db.hir_source_file(source_item_id.file_id).syntax().owned(),
    }
}

pub(crate) fn submodules(
    db: &impl HirDatabase,
    source: ModuleSource,
) -> Cancelable<Arc<Vec<Submodule>>> {
    db.check_canceled()?;
    let file_id = source.file_id();
    let submodules = match source.resolve(db) {
        ModuleSourceNode::SourceFile(it) => collect_submodules(db, file_id, it.borrowed()),
        ModuleSourceNode::Module(it) => it
            .borrowed()
            .item_list()
            .map(|it| collect_submodules(db, file_id, it))
            .unwrap_or_else(Vec::new),
    };
    return Ok(Arc::new(submodules));

    fn collect_submodules<'a>(
        db: &impl HirDatabase,
        file_id: HirFileId,
        root: impl ast::ModuleItemOwner<'a>,
    ) -> Vec<Submodule> {
        modules(root)
            .map(|(name, m)| {
                if m.has_semi() {
                    Submodule::Declaration(name)
                } else {
                    let src = ModuleSource::new_inline(db, file_id, m);
                    Submodule::Definition(name, src)
                }
            })
            .collect()
    }
}

pub(crate) fn modules<'a>(
    root: impl ast::ModuleItemOwner<'a>,
) -> impl Iterator<Item = (Name, ast::Module<'a>)> {
    root.items()
        .filter_map(|item| match item {
            ast::ModuleItem::Module(m) => Some(m),
            _ => None,
        })
        .filter_map(|module| {
            let name = module.name()?.as_name();
            Some((name, module))
        })
}

pub(super) fn input_module_items(
    db: &impl HirDatabase,
    source_root_id: SourceRootId,
    module_id: ModuleId,
) -> Cancelable<Arc<InputModuleItems>> {
    let module_tree = db.module_tree(source_root_id)?;
    let source = module_id.source(&module_tree);
    let file_id = source.file_id();
    let file_items = db.file_items(file_id);
    let fill = |acc: &mut InputModuleItems, items: &mut Iterator<Item = ast::ItemOrMacro>| {
        for item in items {
            match item {
                ast::ItemOrMacro::Item(it) => {
                    acc.add_item(file_id, &file_items, it);
                }
                ast::ItemOrMacro::Macro(macro_call) => {
                    let item_id = file_items.id_of_unchecked(macro_call.syntax());
                    let loc = MacroCallLoc {
                        source_root_id,
                        module_id,
                        source_item_id: SourceItemId {
                            file_id,
                            item_id: Some(item_id),
                        },
                    };
                    let id = loc.id(db);
                    let file_id = HirFileId::from(id);
                    let file_items = db.file_items(file_id);
                    //FIXME: expand recursively
                    for item in db.hir_source_file(file_id).borrowed().items() {
                        acc.add_item(file_id, &file_items, item);
                    }
                }
            }
        }
    };

    let mut res = InputModuleItems::default();
    match source.resolve(db) {
        ModuleSourceNode::SourceFile(it) => fill(&mut res, &mut it.borrowed().items_with_macros()),
        ModuleSourceNode::Module(it) => {
            if let Some(item_list) = it.borrowed().item_list() {
                fill(&mut res, &mut item_list.items_with_macros())
            }
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

    let resolver = Resolver::new(db, &input, source_root, module_tree);
    let res = resolver.resolve()?;
    let elapsed = start.elapsed();
    log::info!("item_map: {:?}", elapsed);
    Ok(Arc::new(res))
}
