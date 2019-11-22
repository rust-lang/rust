//! FIXME: write short doc here

pub(crate) mod src;
pub(crate) mod docs;
pub(crate) mod attrs;

use std::sync::Arc;

use hir_def::{
    adt::VariantData,
    body::scope::ExprScopes,
    builtin_type::BuiltinType,
    data::{ConstData, TraitData},
    nameres::per_ns::PerNs,
    resolver::{HasResolver, TypeNs},
    type_ref::TypeRef,
    ContainerId, CrateModuleId, HasModule, ImplId, LocalEnumVariantId, LocalStructFieldId, Lookup,
    ModuleId, UnionId,
};
use hir_expand::{
    diagnostics::DiagnosticSink,
    name::{self, AsName},
};
use ra_db::{CrateId, Edition};
use ra_syntax::ast;

use crate::{
    db::{DefDatabase, HirDatabase},
    expr::{BindingAnnotation, Body, BodySourceMap, ExprValidator, Pat, PatId},
    ids::{
        AstItemDef, ConstId, EnumId, FunctionId, MacroDefId, StaticId, StructId, TraitId,
        TypeAliasId,
    },
    ty::{InferenceResult, Namespace, TraitRef},
    Either, HasSource, ImportId, Name, Source, Ty,
};

/// hir::Crate describes a single crate. It's the main interface with which
/// a crate's dependencies interact. Mostly, it should be just a proxy for the
/// root module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Crate {
    pub(crate) crate_id: CrateId,
}

#[derive(Debug)]
pub struct CrateDependency {
    pub krate: Crate,
    pub name: Name,
}

impl Crate {
    pub fn crate_id(self) -> CrateId {
        self.crate_id
    }

    pub fn dependencies(self, db: &impl DefDatabase) -> Vec<CrateDependency> {
        db.crate_graph()
            .dependencies(self.crate_id)
            .map(|dep| {
                let krate = Crate { crate_id: dep.crate_id() };
                let name = dep.as_name();
                CrateDependency { krate, name }
            })
            .collect()
    }

    pub fn root_module(self, db: &impl DefDatabase) -> Option<Module> {
        let module_id = db.crate_def_map(self.crate_id).root();
        Some(Module::new(self, module_id))
    }

    pub fn edition(self, db: &impl DefDatabase) -> Edition {
        let crate_graph = db.crate_graph();
        crate_graph.edition(self.crate_id)
    }

    pub fn all(db: &impl DefDatabase) -> Vec<Crate> {
        db.crate_graph().iter().map(|crate_id| Crate { crate_id }).collect()
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

pub use hir_def::ModuleSource;

impl Module {
    pub(crate) fn new(krate: Crate, crate_module_id: CrateModuleId) -> Module {
        Module { id: ModuleId { krate: krate.crate_id, module_id: crate_module_id } }
    }

    /// Name of this module.
    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        let def_map = db.crate_def_map(self.id.krate);
        let parent = def_map[self.id.module_id].parent?;
        def_map[parent].children.iter().find_map(|(name, module_id)| {
            if *module_id == self.id.module_id {
                Some(name.clone())
            } else {
                None
            }
        })
    }

    /// Returns the syntax of the last path segment corresponding to this import
    pub fn import_source(
        self,
        db: &impl HirDatabase,
        import: ImportId,
    ) -> Either<ast::UseTree, ast::ExternCrateItem> {
        let src = self.definition_source(db);
        let (_, source_map) = db.raw_items_with_source_map(src.file_id);
        source_map.get(&src.value, import)
    }

    /// Returns the crate this module is part of.
    pub fn krate(self) -> Crate {
        Crate { crate_id: self.id.krate }
    }

    /// Topmost parent of this module. Every module has a `crate_root`, but some
    /// might be missing `krate`. This can happen if a module's file is not included
    /// in the module tree of any target in `Cargo.toml`.
    pub fn crate_root(self, db: &impl DefDatabase) -> Module {
        let def_map = db.crate_def_map(self.id.krate);
        self.with_module_id(def_map.root())
    }

    /// Finds a child module with the specified name.
    pub fn child(self, db: &impl DefDatabase, name: &Name) -> Option<Module> {
        let def_map = db.crate_def_map(self.id.krate);
        let child_id = def_map[self.id.module_id].children.get(name)?;
        Some(self.with_module_id(*child_id))
    }

    /// Iterates over all child modules.
    pub fn children(self, db: &impl DefDatabase) -> impl Iterator<Item = Module> {
        let def_map = db.crate_def_map(self.id.krate);
        let children = def_map[self.id.module_id]
            .children
            .iter()
            .map(|(_, module_id)| self.with_module_id(*module_id))
            .collect::<Vec<_>>();
        children.into_iter()
    }

    /// Finds a parent module.
    pub fn parent(self, db: &impl DefDatabase) -> Option<Module> {
        let def_map = db.crate_def_map(self.id.krate);
        let parent_id = def_map[self.id.module_id].parent?;
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
    pub fn scope(self, db: &impl HirDatabase) -> Vec<(Name, ScopeDef, Option<ImportId>)> {
        db.crate_def_map(self.id.krate)[self.id.module_id]
            .scope
            .entries()
            .map(|(name, res)| (name.clone(), res.def.into(), res.import))
            .collect()
    }

    pub fn diagnostics(self, db: &impl HirDatabase, sink: &mut DiagnosticSink) {
        db.crate_def_map(self.id.krate).add_diagnostics(db, self.id.module_id, sink);
        for decl in self.declarations(db) {
            match decl {
                crate::ModuleDef::Function(f) => f.diagnostics(db, sink),
                crate::ModuleDef::Module(m) => {
                    // Only add diagnostics from inline modules
                    if let ModuleSource::Module(_) = m.definition_source(db).value {
                        m.diagnostics(db, sink)
                    }
                }
                _ => (),
            }
        }

        for impl_block in self.impl_blocks(db) {
            for item in impl_block.items(db) {
                if let AssocItem::Function(f) = item {
                    f.diagnostics(db, sink);
                }
            }
        }
    }

    pub fn declarations(self, db: &impl DefDatabase) -> Vec<ModuleDef> {
        let def_map = db.crate_def_map(self.id.krate);
        def_map[self.id.module_id].scope.declarations().map(ModuleDef::from).collect()
    }

    pub fn impl_blocks(self, db: &impl DefDatabase) -> Vec<ImplBlock> {
        let def_map = db.crate_def_map(self.id.krate);
        def_map[self.id.module_id].impls.iter().copied().map(ImplBlock::from).collect()
    }

    fn with_module_id(self, module_id: CrateModuleId) -> Module {
        Module::new(self.krate(), module_id)
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
        self.parent.variant_data(db).fields().unwrap()[self.id].name.clone()
    }

    pub fn ty(&self, db: &impl HirDatabase) -> Ty {
        db.type_for_field(*self)
    }

    pub fn parent_def(&self, _db: &impl HirDatabase) -> VariantDef {
        self.parent
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Struct {
    pub(crate) id: StructId,
}

impl Struct {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.0.module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        db.struct_data(self.id.into()).name.clone()
    }

    pub fn fields(self, db: &impl HirDatabase) -> Vec<StructField> {
        db.struct_data(self.id.into())
            .variant_data
            .fields()
            .into_iter()
            .flat_map(|it| it.iter())
            .map(|(id, _)| StructField { parent: self.into(), id })
            .collect()
    }

    pub fn field(self, db: &impl HirDatabase, name: &Name) -> Option<StructField> {
        db.struct_data(self.id.into())
            .variant_data
            .fields()
            .into_iter()
            .flat_map(|it| it.iter())
            .find(|(_id, data)| data.name == *name)
            .map(|(id, _)| StructField { parent: self.into(), id })
    }

    pub fn ty(self, db: &impl HirDatabase) -> Ty {
        db.type_for_def(self.into(), Namespace::Types)
    }

    pub fn constructor_ty(self, db: &impl HirDatabase) -> Ty {
        db.type_for_def(self.into(), Namespace::Values)
    }

    fn variant_data(self, db: &impl DefDatabase) -> Arc<VariantData> {
        db.struct_data(self.id.into()).variant_data.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Union {
    pub(crate) id: UnionId,
}

impl Union {
    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        db.struct_data(self.id.into()).name.clone()
    }

    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.0.module(db) }
    }

    pub fn ty(self, db: &impl HirDatabase) -> Ty {
        db.type_for_def(self.into(), Namespace::Types)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Enum {
    pub(crate) id: EnumId,
}

impl Enum {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        db.enum_data(self.id).name.clone()
    }

    pub fn variants(self, db: &impl DefDatabase) -> Vec<EnumVariant> {
        db.enum_data(self.id)
            .variants
            .iter()
            .map(|(id, _)| EnumVariant { parent: self, id })
            .collect()
    }

    pub fn variant(self, db: &impl DefDatabase, name: &Name) -> Option<EnumVariant> {
        db.enum_data(self.id)
            .variants
            .iter()
            .find(|(_id, data)| data.name.as_ref() == Some(name))
            .map(|(id, _)| EnumVariant { parent: self, id })
    }

    pub fn ty(self, db: &impl HirDatabase) -> Ty {
        db.type_for_def(self.into(), Namespace::Types)
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

    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        db.enum_data(self.parent.id).variants[self.id].name.clone()
    }

    pub fn fields(self, db: &impl HirDatabase) -> Vec<StructField> {
        self.variant_data(db)
            .fields()
            .into_iter()
            .flat_map(|it| it.iter())
            .map(|(id, _)| StructField { parent: self.into(), id })
            .collect()
    }

    pub fn field(self, db: &impl HirDatabase, name: &Name) -> Option<StructField> {
        self.variant_data(db)
            .fields()
            .into_iter()
            .flat_map(|it| it.iter())
            .find(|(_id, data)| data.name == *name)
            .map(|(id, _)| StructField { parent: self.into(), id })
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
    pub fn ty(self, db: &impl HirDatabase) -> Ty {
        match self {
            Adt::Struct(it) => it.ty(db),
            Adt::Union(it) => it.ty(db),
            Adt::Enum(it) => it.ty(db),
        }
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
    EnumVariant(EnumVariant),
}
impl_froms!(VariantDef: Struct, EnumVariant);

impl VariantDef {
    pub fn fields(self, db: &impl HirDatabase) -> Vec<StructField> {
        match self {
            VariantDef::Struct(it) => it.fields(db),
            VariantDef::EnumVariant(it) => it.fields(db),
        }
    }

    pub fn field(self, db: &impl HirDatabase, name: &Name) -> Option<StructField> {
        match self {
            VariantDef::Struct(it) => it.field(db, name),
            VariantDef::EnumVariant(it) => it.field(db, name),
        }
    }

    pub fn module(self, db: &impl HirDatabase) -> Module {
        match self {
            VariantDef::Struct(it) => it.module(db),
            VariantDef::EnumVariant(it) => it.module(db),
        }
    }

    pub(crate) fn variant_data(self, db: &impl DefDatabase) -> Arc<VariantData> {
        match self {
            VariantDef::Struct(it) => it.variant_data(db),
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
    pub(crate) fn krate(self, db: &impl HirDatabase) -> Option<Crate> {
        match self {
            DefWithBody::Const(c) => c.krate(db),
            DefWithBody::Function(f) => f.krate(db),
            DefWithBody::Static(s) => s.krate(db),
        }
    }

    pub fn module(self, db: &impl HirDatabase) -> Module {
        match self {
            DefWithBody::Const(c) => c.module(db),
            DefWithBody::Function(f) => f.module(db),
            DefWithBody::Static(s) => s.module(db),
        }
    }
}

pub trait HasBody: Copy {
    fn infer(self, db: &impl HirDatabase) -> Arc<InferenceResult>;
    fn body(self, db: &impl HirDatabase) -> Arc<Body>;
    fn body_source_map(self, db: &impl HirDatabase) -> Arc<BodySourceMap>;
    fn expr_scopes(self, db: &impl HirDatabase) -> Arc<ExprScopes>;
}

impl<T> HasBody for T
where
    T: Into<DefWithBody> + Copy + HasSource,
{
    fn infer(self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(self.into())
    }

    fn body(self, db: &impl HirDatabase) -> Arc<Body> {
        self.into().body(db)
    }

    fn body_source_map(self, db: &impl HirDatabase) -> Arc<BodySourceMap> {
        self.into().body_source_map(db)
    }

    fn expr_scopes(self, db: &impl HirDatabase) -> Arc<ExprScopes> {
        self.into().expr_scopes(db)
    }
}

impl HasBody for DefWithBody {
    fn infer(self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(self)
    }

    fn body(self, db: &impl HirDatabase) -> Arc<Body> {
        db.body(self.into())
    }

    fn body_source_map(self, db: &impl HirDatabase) -> Arc<BodySourceMap> {
        db.body_with_source_map(self.into()).1
    }

    fn expr_scopes(self, db: &impl HirDatabase) -> Arc<ExprScopes> {
        db.expr_scopes(self.into())
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

    pub(crate) fn body_source_map(self, db: &impl HirDatabase) -> Arc<BodySourceMap> {
        db.body_with_source_map(self.id.into()).1
    }

    pub fn body(self, db: &impl HirDatabase) -> Arc<Body> {
        db.body(self.id.into())
    }

    pub fn ty(self, db: &impl HirDatabase) -> Ty {
        db.type_for_def(self.into(), Namespace::Values)
    }

    pub fn infer(self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(self.into())
    }

    /// The containing impl block, if this is a method.
    pub fn impl_block(self, db: &impl DefDatabase) -> Option<ImplBlock> {
        match self.container(db) {
            Some(Container::ImplBlock(it)) => Some(it),
            _ => None,
        }
    }

    /// The containing trait, if this is a trait method definition.
    pub fn parent_trait(self, db: &impl DefDatabase) -> Option<Trait> {
        match self.container(db) {
            Some(Container::Trait(it)) => Some(it),
            _ => None,
        }
    }

    pub fn container(self, db: &impl DefDatabase) -> Option<Container> {
        match self.id.lookup(db).container {
            ContainerId::TraitId(it) => Some(Container::Trait(it.into())),
            ContainerId::ImplId(it) => Some(Container::ImplBlock(it.into())),
            ContainerId::ModuleId(_) => None,
        }
    }

    pub fn diagnostics(self, db: &impl HirDatabase, sink: &mut DiagnosticSink) {
        let infer = self.infer(db);
        infer.add_diagnostics(db, self, sink);
        let mut validator = ExprValidator::new(self, infer, sink);
        validator.validate_body(db);
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

    pub fn data(self, db: &impl HirDatabase) -> Arc<ConstData> {
        db.const_data(self.id)
    }

    pub fn name(self, db: &impl HirDatabase) -> Option<Name> {
        self.data(db).name.clone()
    }

    pub fn infer(self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(self.into())
    }

    /// The containing impl block, if this is a type alias.
    pub fn impl_block(self, db: &impl DefDatabase) -> Option<ImplBlock> {
        match self.container(db) {
            Some(Container::ImplBlock(it)) => Some(it),
            _ => None,
        }
    }

    /// The containing trait, if this is a trait type alias definition.
    pub fn parent_trait(self, db: &impl DefDatabase) -> Option<Trait> {
        match self.container(db) {
            Some(Container::Trait(it)) => Some(it),
            _ => None,
        }
    }

    pub fn container(self, db: &impl DefDatabase) -> Option<Container> {
        match self.id.lookup(db).container {
            ContainerId::TraitId(it) => Some(Container::Trait(it.into())),
            ContainerId::ImplId(it) => Some(Container::ImplBlock(it.into())),
            ContainerId::ModuleId(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Static {
    pub(crate) id: StaticId,
}

impl Static {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    pub fn data(self, db: &impl HirDatabase) -> Arc<ConstData> {
        db.static_data(self.id)
    }

    pub fn infer(self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(self.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Trait {
    pub(crate) id: TraitId,
}

impl Trait {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.module(db) }
    }

    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        self.trait_data(db).name.clone()
    }

    pub fn items(self, db: &impl DefDatabase) -> Vec<AssocItem> {
        self.trait_data(db).items.iter().map(|it| (*it).into()).collect()
    }

    fn direct_super_traits(self, db: &impl HirDatabase) -> Vec<Trait> {
        let resolver = self.id.resolver(db);
        // returning the iterator directly doesn't easily work because of
        // lifetime problems, but since there usually shouldn't be more than a
        // few direct traits this should be fine (we could even use some kind of
        // SmallVec if performance is a concern)
        db.generic_params(self.id.into())
            .where_predicates
            .iter()
            .filter_map(|pred| match &pred.type_ref {
                TypeRef::Path(p) if p.as_ident() == Some(&name::SELF_TYPE) => pred.bound.as_path(),
                _ => None,
            })
            .filter_map(|path| match resolver.resolve_path_in_type_ns_fully(db, path) {
                Some(TypeNs::TraitId(t)) => Some(t),
                _ => None,
            })
            .map(Trait::from)
            .collect()
    }

    /// Returns an iterator over the whole super trait hierarchy (including the
    /// trait itself).
    pub fn all_super_traits(self, db: &impl HirDatabase) -> Vec<Trait> {
        // we need to take care a bit here to avoid infinite loops in case of cycles
        // (i.e. if we have `trait A: B; trait B: A;`)
        let mut result = vec![self];
        let mut i = 0;
        while i < result.len() {
            let t = result[i];
            // yeah this is quadratic, but trait hierarchies should be flat
            // enough that this doesn't matter
            for tt in t.direct_super_traits(db) {
                if !result.contains(&tt) {
                    result.push(tt);
                }
            }
            i += 1;
        }
        result
    }

    pub fn associated_type_by_name(self, db: &impl DefDatabase, name: &Name) -> Option<TypeAlias> {
        let trait_data = self.trait_data(db);
        let res =
            trait_data.associated_types().map(TypeAlias::from).find(|t| &t.name(db) == name)?;
        Some(res)
    }

    pub fn associated_type_by_name_including_super_traits(
        self,
        db: &impl HirDatabase,
        name: &Name,
    ) -> Option<TypeAlias> {
        self.all_super_traits(db).into_iter().find_map(|t| t.associated_type_by_name(db, name))
    }

    pub(crate) fn trait_data(self, db: &impl DefDatabase) -> Arc<TraitData> {
        db.trait_data(self.id)
    }

    pub fn trait_ref(self, db: &impl HirDatabase) -> TraitRef {
        TraitRef::for_trait(db, self)
    }

    pub fn is_auto(self, db: &impl DefDatabase) -> bool {
        self.trait_data(db).auto
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeAlias {
    pub(crate) id: TypeAliasId,
}

impl TypeAlias {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.lookup(db).module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    /// The containing impl block, if this is a type alias.
    pub fn impl_block(self, db: &impl DefDatabase) -> Option<ImplBlock> {
        match self.container(db) {
            Some(Container::ImplBlock(it)) => Some(it),
            _ => None,
        }
    }

    /// The containing trait, if this is a trait type alias definition.
    pub fn parent_trait(self, db: &impl DefDatabase) -> Option<Trait> {
        match self.container(db) {
            Some(Container::Trait(it)) => Some(it),
            _ => None,
        }
    }

    pub fn container(self, db: &impl DefDatabase) -> Option<Container> {
        match self.id.lookup(db).container {
            ContainerId::TraitId(it) => Some(Container::Trait(it.into())),
            ContainerId::ImplId(it) => Some(Container::ImplBlock(it.into())),
            ContainerId::ModuleId(_) => None,
        }
    }

    pub fn type_ref(self, db: &impl DefDatabase) -> Option<TypeRef> {
        db.type_alias_data(self.id).type_ref.clone()
    }

    pub fn ty(self, db: &impl HirDatabase) -> Ty {
        db.type_for_def(self.into(), Namespace::Types)
    }

    pub fn name(self, db: &impl DefDatabase) -> Name {
        db.type_alias_data(self.id).name.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroDef {
    pub(crate) id: MacroDefId,
}

impl MacroDef {}

pub enum Container {
    Trait(Trait),
    ImplBlock(ImplBlock),
}
impl_froms!(Container: Trait, ImplBlock);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AssocItem {
    Function(Function),
    Const(Const),
    TypeAlias(TypeAlias),
}
// FIXME: not every function, ... is actually an assoc item. maybe we should make
// sure that you can only turn actual assoc items into AssocItems. This would
// require not implementing From, and instead having some checked way of
// casting them, and somehow making the constructors private, which would be annoying.
impl_froms!(AssocItem: Function, Const, TypeAlias);

impl AssocItem {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        match self {
            AssocItem::Function(f) => f.module(db),
            AssocItem::Const(c) => c.module(db),
            AssocItem::TypeAlias(t) => t.module(db),
        }
    }

    pub fn container(self, db: &impl DefDatabase) -> Container {
        match self {
            AssocItem::Function(f) => f.container(db),
            AssocItem::Const(c) => c.container(db),
            AssocItem::TypeAlias(t) => t.container(db),
        }
        .expect("AssocItem without container")
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GenericDef {
    Function(Function),
    Adt(Adt),
    Trait(Trait),
    TypeAlias(TypeAlias),
    ImplBlock(ImplBlock),
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
    ImplBlock,
    EnumVariant,
    Const
);

impl From<AssocItem> for GenericDef {
    fn from(item: AssocItem) -> Self {
        match item {
            AssocItem::Function(f) => f.into(),
            AssocItem::Const(c) => c.into(),
            AssocItem::TypeAlias(t) => t.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Local {
    pub(crate) parent: DefWithBody,
    pub(crate) pat_id: PatId,
}

impl Local {
    pub fn name(self, db: &impl HirDatabase) -> Option<Name> {
        let body = self.parent.body(db);
        match &body[self.pat_id] {
            Pat::Bind { name, .. } => Some(name.clone()),
            _ => None,
        }
    }

    pub fn is_self(self, db: &impl HirDatabase) -> bool {
        self.name(db) == Some(name::SELF_PARAM)
    }

    pub fn is_mut(self, db: &impl HirDatabase) -> bool {
        let body = self.parent.body(db);
        match &body[self.pat_id] {
            Pat::Bind { mode, .. } => match mode {
                BindingAnnotation::Mutable | BindingAnnotation::RefMut => true,
                _ => false,
            },
            _ => false,
        }
    }

    pub fn parent(self, _db: &impl HirDatabase) -> DefWithBody {
        self.parent
    }

    pub fn module(self, db: &impl HirDatabase) -> Module {
        self.parent.module(db)
    }

    pub fn ty(self, db: &impl HirDatabase) -> Ty {
        let infer = db.infer(self.parent);
        infer[self.pat_id].clone()
    }

    pub fn source(self, db: &impl HirDatabase) -> Source<Either<ast::BindPat, ast::SelfParam>> {
        let source_map = self.parent.body_source_map(db);
        let src = source_map.pat_syntax(self.pat_id).unwrap(); // Hmm...
        let root = src.file_syntax(db);
        src.map(|ast| ast.map(|it| it.cast().unwrap().to_node(&root), |it| it.to_node(&root)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GenericParam {
    pub(crate) parent: GenericDef,
    pub(crate) idx: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImplBlock {
    pub(crate) id: ImplId,
}

/// For IDE only
pub enum ScopeDef {
    ModuleDef(ModuleDef),
    MacroDef(MacroDef),
    GenericParam(GenericParam),
    ImplSelfType(ImplBlock),
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
                def.get_macros().map(|macro_def_id| ScopeDef::MacroDef(macro_def_id.into()))
            })
            .unwrap_or(ScopeDef::Unknown)
    }
}
