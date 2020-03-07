//! FIXME: write short doc here
use std::sync::Arc;

use either::Either;
use hir_def::{
    adt::StructKind,
    adt::VariantData,
    builtin_type::BuiltinType,
    docs::Documentation,
    expr::{BindingAnnotation, Pat, PatId},
    per_ns::PerNs,
    resolver::HasResolver,
    type_ref::{Mutability, TypeRef},
    AdtId, AssocContainerId, ConstId, DefWithBodyId, EnumId, FunctionId, GenericDefId, HasModule,
    ImplId, LocalEnumVariantId, LocalModuleId, LocalStructFieldId, Lookup, ModuleId, StaticId,
    StructId, TraitId, TypeAliasId, TypeParamId, UnionId,
};
use hir_expand::{
    diagnostics::DiagnosticSink,
    name::{name, AsName},
    MacroDefId,
};
use hir_ty::{
    autoderef, display::HirFormatter, expr::ExprValidator, method_resolution, ApplicationTy,
    Canonical, InEnvironment, Substs, TraitEnvironment, Ty, TyDefId, TypeCtor,
};
use ra_db::{CrateId, Edition, FileId};
use ra_prof::profile;
use ra_syntax::{
    ast::{self, AttrsOwner, NameOwner},
    AstNode,
};
use rustc_hash::FxHashSet;

use crate::{
    db::{DefDatabase, HirDatabase},
    has_source::HasSource,
    CallableDef, HirDisplay, InFile, Name,
};

/// hir::Crate describes a single crate. It's the main interface with which
/// a crate's dependencies interact. Mostly, it should be just a proxy for the
/// root module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Crate {
    pub(crate) id: CrateId,
}

#[derive(Debug)]
pub struct CrateDependency {
    pub krate: Crate,
    pub name: Name,
}

impl Crate {
    pub fn dependencies(self, db: &impl DefDatabase) -> Vec<CrateDependency> {
        db.crate_graph()
            .dependencies(self.id)
            .map(|dep| {
                let krate = Crate { id: dep.crate_id() };
                let name = dep.as_name();
                CrateDependency { krate, name }
            })
            .collect()
    }

    // FIXME: add `transitive_reverse_dependencies`.
    pub fn reverse_dependencies(self, db: &impl DefDatabase) -> Vec<Crate> {
        let crate_graph = db.crate_graph();
        crate_graph
            .iter()
            .filter(|&krate| crate_graph.dependencies(krate).any(|it| it.crate_id == self.id))
            .map(|id| Crate { id })
            .collect()
    }

    pub fn root_module(self, db: &impl DefDatabase) -> Option<Module> {
        let module_id = db.crate_def_map(self.id).root;
        Some(Module::new(self, module_id))
    }

    pub fn root_file(self, db: &impl DefDatabase) -> FileId {
        db.crate_graph().crate_root(self.id)
    }

    pub fn edition(self, db: &impl DefDatabase) -> Edition {
        let crate_graph = db.crate_graph();
        crate_graph.edition(self.id)
    }

    pub fn all(db: &impl DefDatabase) -> Vec<Crate> {
        db.crate_graph().iter().map(|id| Crate { id }).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Module {
    pub(crate) id: ModuleId,
}

/// The defs which can be visible in the module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModuleDef {
    Module(Module),
    Function(Function),
    Adt(Adt),
    // Can't be directly declared, but can be imported.
    EnumVariant(EnumVariant),
    Const(Const),
    Static(Static),
    Trait(Trait),
    TypeAlias(TypeAlias),
    BuiltinType(BuiltinType),
}
impl_froms!(
    ModuleDef: Module,
    Function,
    Adt(Struct, Enum, Union),
    EnumVariant,
    Const,
    Static,
    Trait,
    TypeAlias,
    BuiltinType
);

impl ModuleDef {
    pub fn module(self, db: &impl HirDatabase) -> Option<Module> {
        match self {
            ModuleDef::Module(it) => it.parent(db),
            ModuleDef::Function(it) => Some(it.module(db)),
            ModuleDef::Adt(it) => Some(it.module(db)),
            ModuleDef::EnumVariant(it) => Some(it.module(db)),
            ModuleDef::Const(it) => Some(it.module(db)),
            ModuleDef::Static(it) => Some(it.module(db)),
            ModuleDef::Trait(it) => Some(it.module(db)),
            ModuleDef::TypeAlias(it) => Some(it.module(db)),
            ModuleDef::BuiltinType(_) => None,
        }
    }
}

pub use hir_def::{
    attr::Attrs, item_scope::ItemInNs, visibility::Visibility, AssocItemId, AssocItemLoc,
};

impl Module {
    pub(crate) fn new(krate: Crate, crate_module_id: LocalModuleId) -> Module {
        Module { id: ModuleId { krate: krate.id, local_id: crate_module_id } }
    }

    /// Name of this module.
    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        let def_map = db.crate_def_map(self.id.krate);
        let parent = def_map[self.id.local_id].parent?;
        def_map[parent].children.iter().find_map(|(name, module_id)| {
            if *module_id == self.id.local_id {
                Some(name.clone())
            } else {
                None
            }
        })
    }

    /// Returns the crate this module is part of.
    pub fn krate(self) -> Crate {
        Crate { id: self.id.krate }
    }

    /// Topmost parent of this module. Every module has a `crate_root`, but some
    /// might be missing `krate`. This can happen if a module's file is not included
    /// in the module tree of any target in `Cargo.toml`.
    pub fn crate_root(self, db: &impl DefDatabase) -> Module {
        let def_map = db.crate_def_map(self.id.krate);
        self.with_module_id(def_map.root)
    }

    /// Iterates over all child modules.
    pub fn children(self, db: &impl DefDatabase) -> impl Iterator<Item = Module> {
        let def_map = db.crate_def_map(self.id.krate);
        let children = def_map[self.id.local_id]
            .children
            .iter()
            .map(|(_, module_id)| self.with_module_id(*module_id))
            .collect::<Vec<_>>();
        children.into_iter()
    }

    /// Finds a parent module.
    pub fn parent(self, db: &impl DefDatabase) -> Option<Module> {
        let def_map = db.crate_def_map(self.id.krate);
        let parent_id = def_map[self.id.local_id].parent?;
        Some(self.with_module_id(parent_id))
    }

    pub fn path_to_root(self, db: &impl HirDatabase) -> Vec<Module> {
        let mut res = vec![self];
        let mut curr = self;
        while let Some(next) = curr.parent(db) {
            res.push(next);
            curr = next
        }
        res
    }

    /// Returns a `ModuleScope`: a set of items, visible in this module.
    pub fn scope(self, db: &impl HirDatabase) -> Vec<(Name, ScopeDef)> {
        db.crate_def_map(self.id.krate)[self.id.local_id]
            .scope
            .entries()
            .map(|(name, def)| (name.clone(), def.into()))
            .collect()
    }

    pub fn diagnostics(self, db: &impl HirDatabase, sink: &mut DiagnosticSink) {
        let _p = profile("Module::diagnostics");
        let crate_def_map = db.crate_def_map(self.id.krate);
        crate_def_map.add_diagnostics(db, self.id.local_id, sink);
        for decl in self.declarations(db) {
            match decl {
                crate::ModuleDef::Function(f) => f.diagnostics(db, sink),
                crate::ModuleDef::Module(m) => {
                    // Only add diagnostics from inline modules
                    if crate_def_map[m.id.local_id].origin.is_inline() {
                        m.diagnostics(db, sink)
                    }
                }
                _ => (),
            }
        }

        for impl_def in self.impl_defs(db) {
            for item in impl_def.items(db) {
                if let AssocItem::Function(f) = item {
                    f.diagnostics(db, sink);
                }
            }
        }
    }

    pub fn declarations(self, db: &impl DefDatabase) -> Vec<ModuleDef> {
        let def_map = db.crate_def_map(self.id.krate);
        def_map[self.id.local_id].scope.declarations().map(ModuleDef::from).collect()
    }

    pub fn impl_defs(self, db: &impl DefDatabase) -> Vec<ImplDef> {
        let def_map = db.crate_def_map(self.id.krate);
        def_map[self.id.local_id].scope.impls().map(ImplDef::from).collect()
    }

    pub(crate) fn with_module_id(self, module_id: LocalModuleId) -> Module {
        Module::new(self.krate(), module_id)
    }

    /// Finds a path that can be used to refer to the given item from within
    /// this module, if possible.
    pub fn find_use_path(
        self,
        db: &impl DefDatabase,
        item: ModuleDef,
    ) -> Option<hir_def::path::ModPath> {
        // FIXME expose namespace choice
        hir_def::find_path::find_path(db, determine_item_namespace(item), self.into())
    }
}

fn determine_item_namespace(module_def: ModuleDef) -> ItemInNs {
    match module_def {
        ModuleDef::Static(_) | ModuleDef::Const(_) | ModuleDef::Function(_) => {
            ItemInNs::Values(module_def.into())
        }
        _ => ItemInNs::Types(module_def.into()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructField {
    pub(crate) parent: VariantDef,
    pub(crate) id: LocalStructFieldId,
}

#[derive(Debug, PartialEq, Eq)]
pub enum FieldSource {
    Named(ast::RecordFieldDef),
    Pos(ast::TupleFieldDef),
}

impl StructField {
    pub fn name(&self, db: &impl HirDatabase) -> Name {
        self.parent.variant_data(db).fields()[self.id].name.clone()
    }

    pub fn ty(&self, db: &impl HirDatabase) -> Type {
        let var_id = self.parent.into();
        let generic_def_id: GenericDefId = match self.parent {
            VariantDef::Struct(it) => it.id.into(),
            VariantDef::Union(it) => it.id.into(),
            VariantDef::EnumVariant(it) => it.parent.id.into(),
        };
        let substs = Substs::type_params(db, generic_def_id);
        let ty = db.field_types(var_id)[self.id].clone().subst(&substs);
        Type::new(db, self.parent.module(db).id.krate, var_id, ty)
    }

    pub fn parent_def(&self, _db: &impl HirDatabase) -> VariantDef {
        self.parent
    }
}

impl HasVisibility for StructField {
    fn visibility(&self, db: &impl HirDatabase) -> Visibility {
        let variant_data = self.parent.variant_data(db);
        let visibility = &variant_data.fields()[self.id].visibility;
        let parent_id: hir_def::VariantId = self.parent.into();
        visibility.resolve(db, &parent_id.resolver(db))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Struct {
    pub(crate) id: StructId,
}

impl Struct {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.lookup(db).container.module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    pub fn name(self, db: &impl DefDatabase) -> Name {
        db.struct_data(self.id).name.clone()
    }

    pub fn fields(self, db: &impl HirDatabase) -> Vec<StructField> {
        db.struct_data(self.id)
            .variant_data
            .fields()
            .iter()
            .map(|(id, _)| StructField { parent: self.into(), id })
            .collect()
    }

    pub fn ty(self, db: &impl HirDatabase) -> Type {
        Type::from_def(db, self.id.lookup(db).container.module(db).krate, self.id)
    }

    fn variant_data(self, db: &impl DefDatabase) -> Arc<VariantData> {
        db.struct_data(self.id).variant_data.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Union {
    pub(crate) id: UnionId,
}

impl Union {
    pub fn name(self, db: &impl DefDatabase) -> Name {
        db.union_data(self.id).name.clone()
    }

    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.lookup(db).container.module(db) }
    }

    pub fn ty(self, db: &impl HirDatabase) -> Type {
        Type::from_def(db, self.id.lookup(db).container.module(db).krate, self.id)
    }

    pub fn fields(self, db: &impl HirDatabase) -> Vec<StructField> {
        db.union_data(self.id)
            .variant_data
            .fields()
            .iter()
            .map(|(id, _)| StructField { parent: self.into(), id })
            .collect()
    }

    fn variant_data(self, db: &impl DefDatabase) -> Arc<VariantData> {
        db.union_data(self.id).variant_data.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Enum {
    pub(crate) id: EnumId,
}

impl Enum {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.lookup(db).container.module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    pub fn name(self, db: &impl DefDatabase) -> Name {
        db.enum_data(self.id).name.clone()
    }

    pub fn variants(self, db: &impl DefDatabase) -> Vec<EnumVariant> {
        db.enum_data(self.id)
            .variants
            .iter()
            .map(|(id, _)| EnumVariant { parent: self, id })
            .collect()
    }

    pub fn ty(self, db: &impl HirDatabase) -> Type {
        Type::from_def(db, self.id.lookup(db).container.module(db).krate, self.id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumVariant {
    pub(crate) parent: Enum,
    pub(crate) id: LocalEnumVariantId,
}

impl EnumVariant {
    pub fn module(self, db: &impl HirDatabase) -> Module {
        self.parent.module(db)
    }
    pub fn parent_enum(self, _db: &impl DefDatabase) -> Enum {
        self.parent
    }

    pub fn name(self, db: &impl DefDatabase) -> Name {
        db.enum_data(self.parent.id).variants[self.id].name.clone()
    }

    pub fn fields(self, db: &impl HirDatabase) -> Vec<StructField> {
        self.variant_data(db)
            .fields()
            .iter()
            .map(|(id, _)| StructField { parent: self.into(), id })
            .collect()
    }

    pub fn kind(self, db: &impl HirDatabase) -> StructKind {
        self.variant_data(db).kind()
    }

    pub(crate) fn variant_data(self, db: &impl DefDatabase) -> Arc<VariantData> {
        db.enum_data(self.parent.id).variants[self.id].variant_data.clone()
    }
}

/// A Data Type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Adt {
    Struct(Struct),
    Union(Union),
    Enum(Enum),
}
impl_froms!(Adt: Struct, Union, Enum);

impl Adt {
    pub fn has_non_default_type_params(self, db: &impl HirDatabase) -> bool {
        let subst = db.generic_defaults(self.into());
        subst.iter().any(|ty| ty == &Ty::Unknown)
    }
    pub fn ty(self, db: &impl HirDatabase) -> Type {
        let id = AdtId::from(self);
        Type::from_def(db, id.module(db).krate, id)
    }

    pub fn module(self, db: &impl DefDatabase) -> Module {
        match self {
            Adt::Struct(s) => s.module(db),
            Adt::Union(s) => s.module(db),
            Adt::Enum(e) => e.module(db),
        }
    }

    pub fn krate(self, db: &impl HirDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VariantDef {
    Struct(Struct),
    Union(Union),
    EnumVariant(EnumVariant),
}
impl_froms!(VariantDef: Struct, Union, EnumVariant);

impl VariantDef {
    pub fn fields(self, db: &impl HirDatabase) -> Vec<StructField> {
        match self {
            VariantDef::Struct(it) => it.fields(db),
            VariantDef::Union(it) => it.fields(db),
            VariantDef::EnumVariant(it) => it.fields(db),
        }
    }

    pub fn module(self, db: &impl HirDatabase) -> Module {
        match self {
            VariantDef::Struct(it) => it.module(db),
            VariantDef::Union(it) => it.module(db),
            VariantDef::EnumVariant(it) => it.module(db),
        }
    }

    pub(crate) fn variant_data(self, db: &impl DefDatabase) -> Arc<VariantData> {
        match self {
            VariantDef::Struct(it) => it.variant_data(db),
            VariantDef::Union(it) => it.variant_data(db),
            VariantDef::EnumVariant(it) => it.variant_data(db),
        }
    }
}

/// The defs which have a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefWithBody {
    Function(Function),
    Static(Static),
    Const(Const),
}

impl_froms!(DefWithBody: Function, Const, Static);

impl DefWithBody {
    pub fn module(self, db: &impl HirDatabase) -> Module {
        match self {
            DefWithBody::Const(c) => c.module(db),
            DefWithBody::Function(f) => f.module(db),
            DefWithBody::Static(s) => s.module(db),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Function {
    pub(crate) id: FunctionId,
}

impl Function {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        self.id.lookup(db).module(db).into()
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    pub fn name(self, db: &impl HirDatabase) -> Name {
        db.function_data(self.id).name.clone()
    }

    pub fn has_self_param(self, db: &impl HirDatabase) -> bool {
        db.function_data(self.id).has_self_param
    }

    pub fn params(self, db: &impl HirDatabase) -> Vec<TypeRef> {
        db.function_data(self.id).params.clone()
    }

    pub fn diagnostics(self, db: &impl HirDatabase, sink: &mut DiagnosticSink) {
        let _p = profile("Function::diagnostics");
        let infer = db.infer(self.id.into());
        infer.add_diagnostics(db, self.id, sink);
        let mut validator = ExprValidator::new(self.id, infer, sink);
        validator.validate_body(db);
    }
}

impl HasVisibility for Function {
    fn visibility(&self, db: &impl HirDatabase) -> Visibility {
        let function_data = db.function_data(self.id);
        let visibility = &function_data.visibility;
        visibility.resolve(db, &self.id.resolver(db))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Const {
    pub(crate) id: ConstId,
}

impl Const {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.lookup(db).module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    pub fn name(self, db: &impl HirDatabase) -> Option<Name> {
        db.const_data(self.id).name.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Static {
    pub(crate) id: StaticId,
}

impl Static {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.lookup(db).module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    pub fn name(self, db: &impl HirDatabase) -> Option<Name> {
        db.static_data(self.id).name.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Trait {
    pub(crate) id: TraitId,
}

impl Trait {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.lookup(db).container.module(db) }
    }

    pub fn name(self, db: &impl DefDatabase) -> Name {
        db.trait_data(self.id).name.clone()
    }

    pub fn items(self, db: &impl DefDatabase) -> Vec<AssocItem> {
        db.trait_data(self.id).items.iter().map(|(_name, it)| (*it).into()).collect()
    }

    pub fn is_auto(self, db: &impl DefDatabase) -> bool {
        db.trait_data(self.id).auto
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeAlias {
    pub(crate) id: TypeAliasId,
}

impl TypeAlias {
    pub fn has_non_default_type_params(self, db: &impl HirDatabase) -> bool {
        let subst = db.generic_defaults(self.id.into());
        subst.iter().any(|ty| ty == &Ty::Unknown)
    }

    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.lookup(db).module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    pub fn type_ref(self, db: &impl DefDatabase) -> Option<TypeRef> {
        db.type_alias_data(self.id).type_ref.clone()
    }

    pub fn ty(self, db: &impl HirDatabase) -> Type {
        Type::from_def(db, self.id.lookup(db).module(db).krate, self.id)
    }

    pub fn name(self, db: &impl DefDatabase) -> Name {
        db.type_alias_data(self.id).name.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroDef {
    pub(crate) id: MacroDefId,
}

impl MacroDef {
    /// FIXME: right now, this just returns the root module of the crate that
    /// defines this macro. The reasons for this is that macros are expanded
    /// early, in `ra_hir_expand`, where modules simply do not exist yet.
    pub fn module(self, db: &impl HirDatabase) -> Option<Module> {
        let krate = self.id.krate?;
        let module_id = db.crate_def_map(krate).root;
        Some(Module::new(Crate { id: krate }, module_id))
    }

    /// XXX: this parses the file
    pub fn name(self, db: &impl HirDatabase) -> Option<Name> {
        self.source(db).value.name().map(|it| it.as_name())
    }
}

/// Invariant: `inner.as_assoc_item(db).is_some()`
/// We do not actively enforce this invariant.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AssocItem {
    Function(Function),
    Const(Const),
    TypeAlias(TypeAlias),
}
pub enum AssocItemContainer {
    Trait(Trait),
    ImplDef(ImplDef),
}
pub trait AsAssocItem {
    fn as_assoc_item(self, db: &impl DefDatabase) -> Option<AssocItem>;
}

impl AsAssocItem for Function {
    fn as_assoc_item(self, db: &impl DefDatabase) -> Option<AssocItem> {
        as_assoc_item(db, AssocItem::Function, self.id)
    }
}
impl AsAssocItem for Const {
    fn as_assoc_item(self, db: &impl DefDatabase) -> Option<AssocItem> {
        as_assoc_item(db, AssocItem::Const, self.id)
    }
}
impl AsAssocItem for TypeAlias {
    fn as_assoc_item(self, db: &impl DefDatabase) -> Option<AssocItem> {
        as_assoc_item(db, AssocItem::TypeAlias, self.id)
    }
}
fn as_assoc_item<ID, DEF, CTOR, AST>(db: &impl DefDatabase, ctor: CTOR, id: ID) -> Option<AssocItem>
where
    ID: Lookup<Data = AssocItemLoc<AST>>,
    DEF: From<ID>,
    CTOR: FnOnce(DEF) -> AssocItem,
    AST: AstNode,
{
    match id.lookup(db).container {
        AssocContainerId::TraitId(_) | AssocContainerId::ImplId(_) => Some(ctor(DEF::from(id))),
        AssocContainerId::ContainerId(_) => None,
    }
}

impl AssocItem {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        match self {
            AssocItem::Function(f) => f.module(db),
            AssocItem::Const(c) => c.module(db),
            AssocItem::TypeAlias(t) => t.module(db),
        }
    }
    pub fn container(self, db: &impl DefDatabase) -> AssocItemContainer {
        let container = match self {
            AssocItem::Function(it) => it.id.lookup(db).container,
            AssocItem::Const(it) => it.id.lookup(db).container,
            AssocItem::TypeAlias(it) => it.id.lookup(db).container,
        };
        match container {
            AssocContainerId::TraitId(id) => AssocItemContainer::Trait(id.into()),
            AssocContainerId::ImplId(id) => AssocItemContainer::ImplDef(id.into()),
            AssocContainerId::ContainerId(_) => panic!("invalid AssocItem"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GenericDef {
    Function(Function),
    Adt(Adt),
    Trait(Trait),
    TypeAlias(TypeAlias),
    ImplDef(ImplDef),
    // enum variants cannot have generics themselves, but their parent enums
    // can, and this makes some code easier to write
    EnumVariant(EnumVariant),
    // consts can have type parameters from their parents (i.e. associated consts of traits)
    Const(Const),
}
impl_froms!(
    GenericDef: Function,
    Adt(Struct, Enum, Union),
    Trait,
    TypeAlias,
    ImplDef,
    EnumVariant,
    Const
);

impl GenericDef {
    pub fn params(self, db: &impl HirDatabase) -> Vec<TypeParam> {
        let generics: Arc<hir_def::generics::GenericParams> = db.generic_params(self.into());
        generics
            .types
            .iter()
            .map(|(local_id, _)| TypeParam { id: TypeParamId { parent: self.into(), local_id } })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Local {
    pub(crate) parent: DefWithBodyId,
    pub(crate) pat_id: PatId,
}

impl Local {
    // FIXME: why is this an option? It shouldn't be?
    pub fn name(self, db: &impl HirDatabase) -> Option<Name> {
        let body = db.body(self.parent.into());
        match &body[self.pat_id] {
            Pat::Bind { name, .. } => Some(name.clone()),
            _ => None,
        }
    }

    pub fn is_self(self, db: &impl HirDatabase) -> bool {
        self.name(db) == Some(name![self])
    }

    pub fn is_mut(self, db: &impl HirDatabase) -> bool {
        let body = db.body(self.parent.into());
        match &body[self.pat_id] {
            Pat::Bind { mode, .. } => match mode {
                BindingAnnotation::Mutable | BindingAnnotation::RefMut => true,
                _ => false,
            },
            _ => false,
        }
    }

    pub fn parent(self, _db: &impl HirDatabase) -> DefWithBody {
        self.parent.into()
    }

    pub fn module(self, db: &impl HirDatabase) -> Module {
        self.parent(db).module(db)
    }

    pub fn ty(self, db: &impl HirDatabase) -> Type {
        let def = DefWithBodyId::from(self.parent);
        let infer = db.infer(def);
        let ty = infer[self.pat_id].clone();
        let resolver = def.resolver(db);
        let krate = def.module(db).krate;
        let environment = TraitEnvironment::lower(db, &resolver);
        Type { krate, ty: InEnvironment { value: ty, environment } }
    }

    pub fn source(self, db: &impl HirDatabase) -> InFile<Either<ast::BindPat, ast::SelfParam>> {
        let (_body, source_map) = db.body_with_source_map(self.parent.into());
        let src = source_map.pat_syntax(self.pat_id).unwrap(); // Hmm...
        let root = src.file_syntax(db);
        src.map(|ast| {
            ast.map_left(|it| it.cast().unwrap().to_node(&root)).map_right(|it| it.to_node(&root))
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeParam {
    pub(crate) id: TypeParamId,
}

impl TypeParam {
    pub fn name(self, db: &impl HirDatabase) -> Name {
        let params = db.generic_params(self.id.parent);
        params.types[self.id.local_id].name.clone().unwrap_or_else(Name::missing)
    }

    pub fn module(self, db: &impl HirDatabase) -> Module {
        self.id.parent.module(db).into()
    }
}

// FIXME: rename from `ImplDef` to `Impl`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImplDef {
    pub(crate) id: ImplId,
}

impl ImplDef {
    pub fn all_in_crate(db: &impl HirDatabase, krate: Crate) -> Vec<ImplDef> {
        let impls = db.impls_in_crate(krate.id);
        impls.all_impls().map(Self::from).collect()
    }
    pub fn for_trait(db: &impl HirDatabase, krate: Crate, trait_: Trait) -> Vec<ImplDef> {
        let impls = db.impls_in_crate(krate.id);
        impls.lookup_impl_defs_for_trait(trait_.id).map(Self::from).collect()
    }

    pub fn target_trait(&self, db: &impl DefDatabase) -> Option<TypeRef> {
        db.impl_data(self.id).target_trait.clone()
    }

    pub fn target_type(&self, db: &impl DefDatabase) -> TypeRef {
        db.impl_data(self.id).target_type.clone()
    }

    pub fn target_ty(&self, db: &impl HirDatabase) -> Type {
        let impl_data = db.impl_data(self.id);
        let resolver = self.id.resolver(db);
        let ctx = hir_ty::TyLoweringContext::new(db, &resolver);
        let environment = TraitEnvironment::lower(db, &resolver);
        let ty = Ty::from_hir(&ctx, &impl_data.target_type);
        Type {
            krate: self.id.lookup(db).container.module(db).krate,
            ty: InEnvironment { value: ty, environment },
        }
    }

    pub fn items(&self, db: &impl DefDatabase) -> Vec<AssocItem> {
        db.impl_data(self.id).items.iter().map(|it| (*it).into()).collect()
    }

    pub fn is_negative(&self, db: &impl DefDatabase) -> bool {
        db.impl_data(self.id).is_negative
    }

    pub fn module(&self, db: &impl DefDatabase) -> Module {
        self.id.lookup(db).container.module(db).into()
    }

    pub fn krate(&self, db: &impl DefDatabase) -> Crate {
        Crate { id: self.module(db).id.krate }
    }

    pub fn is_builtin_derive(&self, db: &impl DefDatabase) -> Option<InFile<ast::Attr>> {
        let src = self.source(db);
        let item = src.file_id.is_builtin_derive(db)?;
        let hygenic = hir_expand::hygiene::Hygiene::new(db, item.file_id);

        let attr = item
            .value
            .attrs()
            .filter_map(|it| {
                let path = hir_def::path::ModPath::from_src(it.path()?, &hygenic)?;
                if path.as_ident()?.to_string() == "derive" {
                    Some(it)
                } else {
                    None
                }
            })
            .last()?;

        Some(item.with_value(attr))
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Type {
    pub(crate) krate: CrateId,
    pub(crate) ty: InEnvironment<Ty>,
}

impl Type {
    fn new(db: &impl HirDatabase, krate: CrateId, lexical_env: impl HasResolver, ty: Ty) -> Type {
        let resolver = lexical_env.resolver(db);
        let environment = TraitEnvironment::lower(db, &resolver);
        Type { krate, ty: InEnvironment { value: ty, environment } }
    }

    fn from_def(
        db: &impl HirDatabase,
        krate: CrateId,
        def: impl HasResolver + Into<TyDefId> + Into<GenericDefId>,
    ) -> Type {
        let substs = Substs::type_params(db, def);
        let ty = db.ty(def.into()).subst(&substs);
        Type::new(db, krate, def, ty)
    }

    pub fn is_bool(&self) -> bool {
        match &self.ty.value {
            Ty::Apply(a_ty) => match a_ty.ctor {
                TypeCtor::Bool => true,
                _ => false,
            },
            _ => false,
        }
    }

    pub fn is_mutable_reference(&self) -> bool {
        match &self.ty.value {
            Ty::Apply(a_ty) => match a_ty.ctor {
                TypeCtor::Ref(Mutability::Mut) => true,
                _ => false,
            },
            _ => false,
        }
    }

    pub fn is_unknown(&self) -> bool {
        match &self.ty.value {
            Ty::Unknown => true,
            _ => false,
        }
    }

    /// Checks that particular type `ty` implements `std::future::Future`.
    /// This function is used in `.await` syntax completion.
    pub fn impls_future(&self, db: &impl HirDatabase) -> bool {
        let krate = self.krate;

        let std_future_trait =
            db.lang_item(krate, "future_trait".into()).and_then(|it| it.as_trait());
        let std_future_trait = match std_future_trait {
            Some(it) => it,
            None => return false,
        };

        let canonical_ty = Canonical { value: self.ty.value.clone(), num_vars: 0 };
        method_resolution::implements_trait(
            &canonical_ty,
            db,
            self.ty.environment.clone(),
            krate,
            std_future_trait,
        )
    }

    // FIXME: this method is broken, as it doesn't take closures into account.
    pub fn as_callable(&self) -> Option<CallableDef> {
        Some(self.ty.value.as_callable()?.0)
    }

    pub fn contains_unknown(&self) -> bool {
        return go(&self.ty.value);

        fn go(ty: &Ty) -> bool {
            match ty {
                Ty::Unknown => true,
                Ty::Apply(a_ty) => a_ty.parameters.iter().any(go),
                _ => false,
            }
        }
    }

    pub fn fields(&self, db: &impl HirDatabase) -> Vec<(StructField, Type)> {
        if let Ty::Apply(a_ty) = &self.ty.value {
            if let TypeCtor::Adt(AdtId::StructId(s)) = a_ty.ctor {
                let var_def = s.into();
                return db
                    .field_types(var_def)
                    .iter()
                    .map(|(local_id, ty)| {
                        let def = StructField { parent: var_def.into(), id: local_id };
                        let ty = ty.clone().subst(&a_ty.parameters);
                        (def, self.derived(ty))
                    })
                    .collect();
            }
        };
        Vec::new()
    }

    pub fn tuple_fields(&self, _db: &impl HirDatabase) -> Vec<Type> {
        let mut res = Vec::new();
        if let Ty::Apply(a_ty) = &self.ty.value {
            if let TypeCtor::Tuple { .. } = a_ty.ctor {
                for ty in a_ty.parameters.iter() {
                    let ty = ty.clone();
                    res.push(self.derived(ty));
                }
            }
        };
        res
    }

    pub fn variant_fields(
        &self,
        db: &impl HirDatabase,
        def: VariantDef,
    ) -> Vec<(StructField, Type)> {
        // FIXME: check that ty and def match
        match &self.ty.value {
            Ty::Apply(a_ty) => {
                let field_types = db.field_types(def.into());
                def.fields(db)
                    .into_iter()
                    .map(|it| {
                        let ty = field_types[it.id].clone().subst(&a_ty.parameters);
                        (it, self.derived(ty))
                    })
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    pub fn autoderef<'a>(&'a self, db: &'a impl HirDatabase) -> impl Iterator<Item = Type> + 'a {
        // There should be no inference vars in types passed here
        // FIXME check that?
        let canonical = Canonical { value: self.ty.value.clone(), num_vars: 0 };
        let environment = self.ty.environment.clone();
        let ty = InEnvironment { value: canonical, environment };
        autoderef(db, Some(self.krate), ty)
            .map(|canonical| canonical.value)
            .map(move |ty| self.derived(ty))
    }

    // This would be nicer if it just returned an iterator, but that runs into
    // lifetime problems, because we need to borrow temp `CrateImplDefs`.
    pub fn iterate_impl_items<T>(
        self,
        db: &impl HirDatabase,
        krate: Crate,
        mut callback: impl FnMut(AssocItem) -> Option<T>,
    ) -> Option<T> {
        for krate in self.ty.value.def_crates(db, krate.id)? {
            let impls = db.impls_in_crate(krate);

            for impl_def in impls.lookup_impl_defs(&self.ty.value) {
                for &item in db.impl_data(impl_def).items.iter() {
                    if let Some(result) = callback(item.into()) {
                        return Some(result);
                    }
                }
            }
        }
        None
    }

    pub fn iterate_method_candidates<T>(
        &self,
        db: &impl HirDatabase,
        krate: Crate,
        traits_in_scope: &FxHashSet<TraitId>,
        name: Option<&Name>,
        mut callback: impl FnMut(&Ty, Function) -> Option<T>,
    ) -> Option<T> {
        // There should be no inference vars in types passed here
        // FIXME check that?
        // FIXME replace Unknown by bound vars here
        let canonical = Canonical { value: self.ty.value.clone(), num_vars: 0 };

        let env = self.ty.environment.clone();
        let krate = krate.id;

        method_resolution::iterate_method_candidates(
            &canonical,
            db,
            env,
            krate,
            traits_in_scope,
            name,
            method_resolution::LookupMode::MethodCall,
            |ty, it| match it {
                AssocItemId::FunctionId(f) => callback(ty, f.into()),
                _ => None,
            },
        )
    }

    pub fn iterate_path_candidates<T>(
        &self,
        db: &impl HirDatabase,
        krate: Crate,
        traits_in_scope: &FxHashSet<TraitId>,
        name: Option<&Name>,
        mut callback: impl FnMut(&Ty, AssocItem) -> Option<T>,
    ) -> Option<T> {
        // There should be no inference vars in types passed here
        // FIXME check that?
        // FIXME replace Unknown by bound vars here
        let canonical = Canonical { value: self.ty.value.clone(), num_vars: 0 };

        let env = self.ty.environment.clone();
        let krate = krate.id;

        method_resolution::iterate_method_candidates(
            &canonical,
            db,
            env,
            krate,
            traits_in_scope,
            name,
            method_resolution::LookupMode::Path,
            |ty, it| callback(ty, it.into()),
        )
    }

    pub fn as_adt(&self) -> Option<Adt> {
        let (adt, _subst) = self.ty.value.as_adt()?;
        Some(adt.into())
    }

    // FIXME: provide required accessors such that it becomes implementable from outside.
    pub fn is_equal_for_find_impls(&self, other: &Type) -> bool {
        match (&self.ty.value, &other.ty.value) {
            (Ty::Apply(a_original_ty), Ty::Apply(ApplicationTy { ctor, parameters })) => match ctor
            {
                TypeCtor::Ref(..) => match parameters.as_single() {
                    Ty::Apply(a_ty) => a_original_ty.ctor == a_ty.ctor,
                    _ => false,
                },
                _ => a_original_ty.ctor == *ctor,
            },
            _ => false,
        }
    }

    fn derived(&self, ty: Ty) -> Type {
        Type {
            krate: self.krate,
            ty: InEnvironment { value: ty, environment: self.ty.environment.clone() },
        }
    }
}

impl HirDisplay for Type {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> std::fmt::Result {
        self.ty.value.hir_fmt(f)
    }
}

/// For IDE only
pub enum ScopeDef {
    ModuleDef(ModuleDef),
    MacroDef(MacroDef),
    GenericParam(TypeParam),
    ImplSelfType(ImplDef),
    AdtSelfType(Adt),
    Local(Local),
    Unknown,
}

impl From<PerNs> for ScopeDef {
    fn from(def: PerNs) -> Self {
        def.take_types()
            .or_else(|| def.take_values())
            .map(|module_def_id| ScopeDef::ModuleDef(module_def_id.into()))
            .or_else(|| {
                def.take_macros().map(|macro_def_id| ScopeDef::MacroDef(macro_def_id.into()))
            })
            .unwrap_or(ScopeDef::Unknown)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttrDef {
    Module(Module),
    StructField(StructField),
    Adt(Adt),
    Function(Function),
    EnumVariant(EnumVariant),
    Static(Static),
    Const(Const),
    Trait(Trait),
    TypeAlias(TypeAlias),
    MacroDef(MacroDef),
}

impl_froms!(
    AttrDef: Module,
    StructField,
    Adt(Struct, Enum, Union),
    EnumVariant,
    Static,
    Const,
    Function,
    Trait,
    TypeAlias,
    MacroDef
);

pub trait HasAttrs {
    fn attrs(self, db: &impl DefDatabase) -> Attrs;
}

impl<T: Into<AttrDef>> HasAttrs for T {
    fn attrs(self, db: &impl DefDatabase) -> Attrs {
        let def: AttrDef = self.into();
        db.attrs(def.into())
    }
}

pub trait Docs {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation>;
}
impl<T: Into<AttrDef> + Copy> Docs for T {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        let def: AttrDef = (*self).into();
        db.documentation(def.into())
    }
}

pub trait HasVisibility {
    fn visibility(&self, db: &impl HirDatabase) -> Visibility;
    fn is_visible_from(&self, db: &impl HirDatabase, module: Module) -> bool {
        let vis = self.visibility(db);
        vis.is_visible_from(db, module.id)
    }
}
