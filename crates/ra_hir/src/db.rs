use std::sync::Arc;

use ra_syntax::SyntaxNode;
use ra_db::{SourceRootId, LocationIntener, SyntaxDatabase, FileId, Cancelable};

use crate::{
    DefLoc, DefId, Name,
    SourceFileItems, SourceItemId,
    query_definitions,
    FnScopes,
    module::{ModuleId, ModuleTree, ModuleSource,
    nameres::{ItemMap, InputModuleItems}},
    ty::{InferenceResult, Ty},
    adt::{StructData, EnumData},
};

salsa::query_group! {

pub trait HirDatabase: SyntaxDatabase
    + AsRef<LocationIntener<DefLoc, DefId>>
{
    fn fn_scopes(def_id: DefId) -> Arc<FnScopes> {
        type FnScopesQuery;
        use fn query_definitions::fn_scopes;
    }

    fn struct_data(def_id: DefId) -> Cancelable<Arc<StructData>> {
        type StructDataQuery;
        use fn query_definitions::struct_data;
    }

    fn enum_data(def_id: DefId) -> Cancelable<Arc<EnumData>> {
        type EnumDataQuery;
        use fn query_definitions::enum_data;
    }

    fn infer(def_id: DefId) -> Cancelable<Arc<InferenceResult>> {
        type InferQuery;
        use fn crate::ty::infer;
    }

    fn type_for_def(def_id: DefId) -> Cancelable<Ty> {
        type TypeForDefQuery;
        use fn crate::ty::type_for_def;
    }

    fn type_for_field(def_id: DefId, field: Name) -> Cancelable<Ty> {
        type TypeForFieldQuery;
        use fn crate::ty::type_for_field;
    }

    fn file_items(file_id: FileId) -> Arc<SourceFileItems> {
        type SourceFileItemsQuery;
        use fn query_definitions::file_items;
    }

    fn file_item(source_item_id: SourceItemId) -> SyntaxNode {
        type FileItemQuery;
        use fn query_definitions::file_item;
    }

    fn submodules(source: ModuleSource) -> Cancelable<Arc<Vec<crate::module::imp::Submodule>>> {
        type SubmodulesQuery;
        use fn query_definitions::submodules;
    }

    fn input_module_items(source_root_id: SourceRootId, module_id: ModuleId) -> Cancelable<Arc<InputModuleItems>> {
        type InputModuleItemsQuery;
        use fn query_definitions::input_module_items;
    }
    fn item_map(source_root_id: SourceRootId) -> Cancelable<Arc<ItemMap>> {
        type ItemMapQuery;
        use fn query_definitions::item_map;
    }
    fn module_tree(source_root_id: SourceRootId) -> Cancelable<Arc<ModuleTree>> {
        type ModuleTreeQuery;
        use fn crate::module::imp::module_tree;
    }
}

}
