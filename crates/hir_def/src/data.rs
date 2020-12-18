//! Contains basic data about various HIR declarations.

use std::sync::Arc;

use hir_expand::{name::Name, InFile};
use syntax::ast;

use crate::{
    attr::Attrs,
    body::Expander,
    db::DefDatabase,
    item_tree::{AssocItem, ItemTreeId, ModItem},
    type_ref::{TypeBound, TypeRef},
    visibility::RawVisibility,
    AssocContainerId, AssocItemId, ConstId, ConstLoc, FunctionId, FunctionLoc, HasModule, ImplId,
    Intern, Lookup, ModuleId, StaticId, TraitId, TypeAliasId, TypeAliasLoc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionData {
    pub name: Name,
    pub params: Vec<TypeRef>,
    pub ret_type: TypeRef,
    pub attrs: Attrs,
    /// True if the first param is `self`. This is relevant to decide whether this
    /// can be called as a method.
    pub has_self_param: bool,
    pub has_body: bool,
    pub is_unsafe: bool,
    pub is_varargs: bool,
    pub is_extern: bool,
    pub visibility: RawVisibility,
}

impl FunctionData {
    pub(crate) fn fn_data_query(db: &dyn DefDatabase, func: FunctionId) -> Arc<FunctionData> {
        let loc = func.lookup(db);
        let krate = loc.container.module(db).krate;
        let item_tree = db.item_tree(loc.id.file_id);
        let func = &item_tree[loc.id.value];

        Arc::new(FunctionData {
            name: func.name.clone(),
            params: func.params.to_vec(),
            ret_type: func.ret_type.clone(),
            attrs: item_tree.attrs(db, krate, ModItem::from(loc.id.value).into()).clone(),
            has_self_param: func.has_self_param,
            has_body: func.has_body,
            is_unsafe: func.is_unsafe,
            is_varargs: func.is_varargs,
            is_extern: func.is_extern,
            visibility: item_tree[func.visibility].clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAliasData {
    pub name: Name,
    pub type_ref: Option<TypeRef>,
    pub visibility: RawVisibility,
    pub is_extern: bool,
    /// Bounds restricting the type alias itself (eg. `type Ty: Bound;` in a trait or impl).
    pub bounds: Vec<TypeBound>,
}

impl TypeAliasData {
    pub(crate) fn type_alias_data_query(
        db: &dyn DefDatabase,
        typ: TypeAliasId,
    ) -> Arc<TypeAliasData> {
        let loc = typ.lookup(db);
        let item_tree = db.item_tree(loc.id.file_id);
        let typ = &item_tree[loc.id.value];

        Arc::new(TypeAliasData {
            name: typ.name.clone(),
            type_ref: typ.type_ref.clone(),
            visibility: item_tree[typ.visibility].clone(),
            is_extern: typ.is_extern,
            bounds: typ.bounds.to_vec(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitData {
    pub name: Name,
    pub items: Vec<(Name, AssocItemId)>,
    pub auto: bool,
}

impl TraitData {
    pub(crate) fn trait_data_query(db: &dyn DefDatabase, tr: TraitId) -> Arc<TraitData> {
        let tr_loc = tr.lookup(db);
        let item_tree = db.item_tree(tr_loc.id.file_id);
        let tr_def = &item_tree[tr_loc.id.value];
        let name = tr_def.name.clone();
        let auto = tr_def.auto;
        let module_id = tr_loc.container.module(db);
        let container = AssocContainerId::TraitId(tr);
        let mut expander = Expander::new(db, tr_loc.id.file_id, module_id);

        let items = collect_items(
            db,
            module_id,
            &mut expander,
            tr_def.items.iter().copied(),
            tr_loc.id.file_id,
            container,
            100,
        );

        Arc::new(TraitData { name, items, auto })
    }

    pub fn associated_types(&self) -> impl Iterator<Item = TypeAliasId> + '_ {
        self.items.iter().filter_map(|(_name, item)| match item {
            AssocItemId::TypeAliasId(t) => Some(*t),
            _ => None,
        })
    }

    pub fn associated_type_by_name(&self, name: &Name) -> Option<TypeAliasId> {
        self.items.iter().find_map(|(item_name, item)| match item {
            AssocItemId::TypeAliasId(t) if item_name == name => Some(*t),
            _ => None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplData {
    pub target_trait: Option<TypeRef>,
    pub target_type: TypeRef,
    pub items: Vec<AssocItemId>,
    pub is_negative: bool,
}

impl ImplData {
    pub(crate) fn impl_data_query(db: &dyn DefDatabase, id: ImplId) -> Arc<ImplData> {
        let _p = profile::span("impl_data_query");
        let impl_loc = id.lookup(db);

        let item_tree = db.item_tree(impl_loc.id.file_id);
        let impl_def = &item_tree[impl_loc.id.value];
        let target_trait = impl_def.target_trait.clone();
        let target_type = impl_def.target_type.clone();
        let is_negative = impl_def.is_negative;
        let module_id = impl_loc.container.module(db);
        let container = AssocContainerId::ImplId(id);
        let mut expander = Expander::new(db, impl_loc.id.file_id, module_id);

        let items = collect_items(
            db,
            module_id,
            &mut expander,
            impl_def.items.iter().copied(),
            impl_loc.id.file_id,
            container,
            100,
        );
        let items = items.into_iter().map(|(_, item)| item).collect();

        Arc::new(ImplData { target_trait, target_type, items, is_negative })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstData {
    /// const _: () = ();
    pub name: Option<Name>,
    pub type_ref: TypeRef,
    pub visibility: RawVisibility,
}

impl ConstData {
    pub(crate) fn const_data_query(db: &dyn DefDatabase, konst: ConstId) -> Arc<ConstData> {
        let loc = konst.lookup(db);
        let item_tree = db.item_tree(loc.id.file_id);
        let konst = &item_tree[loc.id.value];

        Arc::new(ConstData {
            name: konst.name.clone(),
            type_ref: konst.type_ref.clone(),
            visibility: item_tree[konst.visibility].clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticData {
    pub name: Option<Name>,
    pub type_ref: TypeRef,
    pub visibility: RawVisibility,
    pub mutable: bool,
    pub is_extern: bool,
}

impl StaticData {
    pub(crate) fn static_data_query(db: &dyn DefDatabase, konst: StaticId) -> Arc<StaticData> {
        let node = konst.lookup(db);
        let item_tree = db.item_tree(node.id.file_id);
        let statik = &item_tree[node.id.value];

        Arc::new(StaticData {
            name: Some(statik.name.clone()),
            type_ref: statik.type_ref.clone(),
            visibility: item_tree[statik.visibility].clone(),
            mutable: statik.mutable,
            is_extern: statik.is_extern,
        })
    }
}

fn collect_items(
    db: &dyn DefDatabase,
    module: ModuleId,
    expander: &mut Expander,
    assoc_items: impl Iterator<Item = AssocItem>,
    file_id: crate::HirFileId,
    container: AssocContainerId,
    limit: usize,
) -> Vec<(Name, AssocItemId)> {
    if limit == 0 {
        return Vec::new();
    }

    let item_tree = db.item_tree(file_id);
    let cfg_options = db.crate_graph()[module.krate].cfg_options.clone();

    let mut items = Vec::new();
    for item in assoc_items {
        match item {
            AssocItem::Function(id) => {
                let item = &item_tree[id];
                let attrs = item_tree.attrs(db, module.krate, ModItem::from(id).into());
                if !attrs.is_cfg_enabled(&cfg_options) {
                    continue;
                }
                let def = FunctionLoc { container, id: ItemTreeId::new(file_id, id) }.intern(db);
                items.push((item.name.clone(), def.into()));
            }
            // FIXME: cfg?
            AssocItem::Const(id) => {
                let item = &item_tree[id];
                let name = match item.name.clone() {
                    Some(name) => name,
                    None => continue,
                };
                let def = ConstLoc { container, id: ItemTreeId::new(file_id, id) }.intern(db);
                items.push((name, def.into()));
            }
            AssocItem::TypeAlias(id) => {
                let item = &item_tree[id];
                let def = TypeAliasLoc { container, id: ItemTreeId::new(file_id, id) }.intern(db);
                items.push((item.name.clone(), def.into()));
            }
            AssocItem::MacroCall(call) => {
                let call = &item_tree[call];
                let ast_id_map = db.ast_id_map(file_id);
                let root = db.parse_or_expand(file_id).unwrap();
                let call = ast_id_map.get(call.ast_id).to_node(&root);

                if let Some((mark, mac)) = expander.enter_expand(db, None, call).value {
                    let src: InFile<ast::MacroItems> = expander.to_source(mac);
                    let item_tree = db.item_tree(src.file_id);
                    let iter =
                        item_tree.top_level_items().iter().filter_map(ModItem::as_assoc_item);
                    items.extend(collect_items(
                        db,
                        module,
                        expander,
                        iter,
                        src.file_id,
                        container,
                        limit - 1,
                    ));

                    expander.exit(db, mark);
                }
            }
        }
    }

    items
}
