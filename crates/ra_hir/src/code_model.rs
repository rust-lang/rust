//! FIXME: write short doc here

pub(crate) mod src;
pub(crate) mod docs;

use std::sync::Arc;

use hir_def::{
    adt::VariantData,
    builtin_type::BuiltinType,
    type_ref::{Mutability, TypeRef},
    CrateModuleId, LocalEnumVariantId, LocalStructFieldId, ModuleId, UnionId,
};
use hir_expand::{
    diagnostics::DiagnosticSink,
    name::{self, AsName},
};
use ra_db::{CrateId, Edition};
use ra_syntax::ast::{self, NameOwner, TypeAscriptionOwner};

use crate::{
    adt::VariantDef,
    db::{AstDatabase, DefDatabase, HirDatabase},
    expr::{validation::ExprValidator, Body, BodySourceMap},
    generics::HasGenericParams,
    ids::{
        AstItemDef, ConstId, EnumId, FunctionId, MacroDefId, StaticId, StructId, TraitId,
        TypeAliasId,
    },
    impl_block::ImplBlock,
    resolve::{Resolver, Scope, TypeNs},
    traits::TraitData,
    ty::{InferenceResult, TraitRef},
    Either, HasSource, Name, ScopeDef, Ty, {ImportId, Namespace},
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
        source_map.get(&src.ast, import)
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
    pub fn child(self, db: &impl HirDatabase, name: &Name) -> Option<Module> {
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
                    if let ModuleSource::Module(_) = m.definition_source(db).ast {
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

    pub(crate) fn resolver(self, db: &impl DefDatabase) -> Resolver {
        let def_map = db.crate_def_map(self.id.krate);
        Resolver::default().push_module_scope(def_map, self.id.module_id)
    }

    pub fn declarations(self, db: &impl DefDatabase) -> Vec<ModuleDef> {
        let def_map = db.crate_def_map(self.id.krate);
        def_map[self.id.module_id]
            .scope
            .entries()
            .filter_map(|(_name, res)| if res.import.is_none() { Some(res.def) } else { None })
            .flat_map(|per_ns| {
                per_ns.take_types().into_iter().chain(per_ns.take_values().into_iter())
            })
            .map(ModuleDef::from)
            .collect()
    }

    pub fn impl_blocks(self, db: &impl DefDatabase) -> Vec<ImplBlock> {
        let module_impl_blocks = db.impls_in_module(self);
        module_impl_blocks
            .impls
            .iter()
            .map(|(impl_id, _)| ImplBlock::from_id(self, impl_id))
            .collect()
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
        Module { id: self.id.module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        db.struct_data(self.id).name.clone()
    }

    pub fn fields(self, db: &impl HirDatabase) -> Vec<StructField> {
        db.struct_data(self.id)
            .variant_data
            .fields()
            .into_iter()
            .flat_map(|it| it.iter())
            .map(|(id, _)| StructField { parent: self.into(), id })
            .collect()
    }

    pub fn field(self, db: &impl HirDatabase, name: &Name) -> Option<StructField> {
        db.struct_data(self.id)
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

    // FIXME move to a more general type
    /// Builds a resolver for type references inside this struct.
    pub(crate) fn resolver(self, db: &impl HirDatabase) -> Resolver {
        // take the outer scope...
        let r = self.module(db).resolver(db);
        // ...and add generic params, if present
        let p = self.generic_params(db);
        let r = if !p.params.is_empty() { r.push_generic_params_scope(p) } else { r };
        r
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Union {
    pub(crate) id: UnionId,
}

impl Union {
    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        db.union_data(self.id).name.clone()
    }

    pub fn module(self, db: &impl HirDatabase) -> Module {
        Module { id: self.id.module(db) }
    }

    pub fn ty(self, db: &impl HirDatabase) -> Ty {
        db.type_for_def(self.into(), Namespace::Types)
    }

    // FIXME move to a more general type
    /// Builds a resolver for type references inside this union.
    pub(crate) fn resolver(self, db: &impl HirDatabase) -> Resolver {
        // take the outer scope...
        let r = self.module(db).resolver(db);
        // ...and add generic params, if present
        let p = self.generic_params(db);
        let r = if !p.params.is_empty() { r.push_generic_params_scope(p) } else { r };
        r
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

    // FIXME: move to a more general type
    /// Builds a resolver for type references inside this struct.
    pub(crate) fn resolver(self, db: &impl HirDatabase) -> Resolver {
        // take the outer scope...
        let r = self.module(db).resolver(db);
        // ...and add generic params, if present
        let p = self.generic_params(db);
        let r = if !p.params.is_empty() { r.push_generic_params_scope(p) } else { r };
        r.push_scope(Scope::AdtScope(self.into()))
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

    pub fn krate(self, db: &impl HirDatabase) -> Option<Crate> {
        Some(
            match self {
                Adt::Struct(s) => s.module(db),
                Adt::Union(s) => s.module(db),
                Adt::Enum(e) => e.module(db),
            }
            .krate(),
        )
    }

    pub(crate) fn resolver(self, db: &impl HirDatabase) -> Resolver {
        match self {
            Adt::Struct(it) => it.resolver(db),
            Adt::Union(it) => it.resolver(db),
            Adt::Enum(it) => it.resolver(db),
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
    /// Builds a resolver for code inside this item.
    pub(crate) fn resolver(self, db: &impl HirDatabase) -> Resolver {
        match self {
            DefWithBody::Const(c) => c.resolver(db),
            DefWithBody::Function(f) => f.resolver(db),
            DefWithBody::Static(s) => s.resolver(db),
        }
    }

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
}

impl<T> HasBody for T
where
    T: Into<DefWithBody> + Copy + HasSource,
{
    fn infer(self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(self.into())
    }

    fn body(self, db: &impl HirDatabase) -> Arc<Body> {
        db.body_hir(self.into())
    }

    fn body_source_map(self, db: &impl HirDatabase) -> Arc<BodySourceMap> {
        db.body_with_source_map(self.into()).1
    }
}

impl HasBody for DefWithBody {
    fn infer(self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(self)
    }

    fn body(self, db: &impl HirDatabase) -> Arc<Body> {
        db.body_hir(self)
    }

    fn body_source_map(self, db: &impl HirDatabase) -> Arc<BodySourceMap> {
        db.body_with_source_map(self).1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Function {
    pub(crate) id: FunctionId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnData {
    pub(crate) name: Name,
    pub(crate) params: Vec<TypeRef>,
    pub(crate) ret_type: TypeRef,
    /// True if the first param is `self`. This is relevant to decide whether this
    /// can be called as a method.
    pub(crate) has_self_param: bool,
}

impl FnData {
    pub(crate) fn fn_data_query(
        db: &(impl DefDatabase + AstDatabase),
        func: Function,
    ) -> Arc<FnData> {
        let src = func.source(db);
        let name = src.ast.name().map(|n| n.as_name()).unwrap_or_else(Name::missing);
        let mut params = Vec::new();
        let mut has_self_param = false;
        if let Some(param_list) = src.ast.param_list() {
            if let Some(self_param) = param_list.self_param() {
                let self_type = if let Some(type_ref) = self_param.ascribed_type() {
                    TypeRef::from_ast(type_ref)
                } else {
                    let self_type = TypeRef::Path(name::SELF_TYPE.into());
                    match self_param.kind() {
                        ast::SelfParamKind::Owned => self_type,
                        ast::SelfParamKind::Ref => {
                            TypeRef::Reference(Box::new(self_type), Mutability::Shared)
                        }
                        ast::SelfParamKind::MutRef => {
                            TypeRef::Reference(Box::new(self_type), Mutability::Mut)
                        }
                    }
                };
                params.push(self_type);
                has_self_param = true;
            }
            for param in param_list.params() {
                let type_ref = TypeRef::from_ast_opt(param.ascribed_type());
                params.push(type_ref);
            }
        }
        let ret_type = if let Some(type_ref) = src.ast.ret_type().and_then(|rt| rt.type_ref()) {
            TypeRef::from_ast(type_ref)
        } else {
            TypeRef::unit()
        };

        let sig = FnData { name, params, ret_type, has_self_param };
        Arc::new(sig)
    }
    pub fn name(&self) -> &Name {
        &self.name
    }

    pub fn params(&self) -> &[TypeRef] {
        &self.params
    }

    pub fn ret_type(&self) -> &TypeRef {
        &self.ret_type
    }

    /// True if the first arg is `self`. This is relevant to decide whether this
    /// can be called as a method.
    pub fn has_self_param(&self) -> bool {
        self.has_self_param
    }
}

impl Function {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    pub fn name(self, db: &impl HirDatabase) -> Name {
        self.data(db).name.clone()
    }

    pub(crate) fn body_source_map(self, db: &impl HirDatabase) -> Arc<BodySourceMap> {
        db.body_with_source_map(self.into()).1
    }

    pub fn body(self, db: &impl HirDatabase) -> Arc<Body> {
        db.body_hir(self.into())
    }

    pub fn ty(self, db: &impl HirDatabase) -> Ty {
        db.type_for_def(self.into(), Namespace::Values)
    }

    pub fn data(self, db: &impl HirDatabase) -> Arc<FnData> {
        db.fn_data(self)
    }

    pub fn infer(self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(self.into())
    }

    /// The containing impl block, if this is a method.
    pub fn impl_block(self, db: &impl DefDatabase) -> Option<ImplBlock> {
        let module_impls = db.impls_in_module(self.module(db));
        ImplBlock::containing(module_impls, self.into())
    }

    /// The containing trait, if this is a trait method definition.
    pub fn parent_trait(self, db: &impl DefDatabase) -> Option<Trait> {
        db.trait_items_index(self.module(db)).get_parent_trait(self.into())
    }

    pub fn container(self, db: &impl DefDatabase) -> Option<Container> {
        if let Some(impl_block) = self.impl_block(db) {
            Some(impl_block.into())
        } else if let Some(trait_) = self.parent_trait(db) {
            Some(trait_.into())
        } else {
            None
        }
    }

    // FIXME: move to a more general type for 'body-having' items
    /// Builds a resolver for code inside this item.
    pub(crate) fn resolver(self, db: &impl HirDatabase) -> Resolver {
        // take the outer scope...
        let r = self.container(db).map_or_else(|| self.module(db).resolver(db), |c| c.resolver(db));
        // ...and add generic params, if present
        let p = self.generic_params(db);
        let r = if !p.params.is_empty() { r.push_generic_params_scope(p) } else { r };
        r
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
        Module { id: self.id.module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    pub fn data(self, db: &impl HirDatabase) -> Arc<ConstData> {
        db.const_data(self)
    }

    pub fn name(self, db: &impl HirDatabase) -> Option<Name> {
        self.data(db).name().cloned()
    }

    pub fn infer(self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(self.into())
    }

    /// The containing impl block, if this is a method.
    pub fn impl_block(self, db: &impl DefDatabase) -> Option<ImplBlock> {
        let module_impls = db.impls_in_module(self.module(db));
        ImplBlock::containing(module_impls, self.into())
    }

    pub fn parent_trait(self, db: &impl DefDatabase) -> Option<Trait> {
        db.trait_items_index(self.module(db)).get_parent_trait(self.into())
    }

    pub fn container(self, db: &impl DefDatabase) -> Option<Container> {
        if let Some(impl_block) = self.impl_block(db) {
            Some(impl_block.into())
        } else if let Some(trait_) = self.parent_trait(db) {
            Some(trait_.into())
        } else {
            None
        }
    }

    // FIXME: move to a more general type for 'body-having' items
    /// Builds a resolver for code inside this item.
    pub(crate) fn resolver(self, db: &impl HirDatabase) -> Resolver {
        // take the outer scope...
        let r = self
            .impl_block(db)
            .map(|ib| ib.resolver(db))
            .unwrap_or_else(|| self.module(db).resolver(db));
        r
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstData {
    pub(crate) name: Option<Name>,
    pub(crate) type_ref: TypeRef,
}

impl ConstData {
    pub fn name(&self) -> Option<&Name> {
        self.name.as_ref()
    }

    pub fn type_ref(&self) -> &TypeRef {
        &self.type_ref
    }

    pub(crate) fn const_data_query(
        db: &(impl DefDatabase + AstDatabase),
        konst: Const,
    ) -> Arc<ConstData> {
        let node = konst.source(db).ast;
        const_data_for(&node)
    }

    pub(crate) fn static_data_query(
        db: &(impl DefDatabase + AstDatabase),
        konst: Static,
    ) -> Arc<ConstData> {
        let node = konst.source(db).ast;
        const_data_for(&node)
    }
}

fn const_data_for<N: NameOwner + TypeAscriptionOwner>(node: &N) -> Arc<ConstData> {
    let name = node.name().map(|n| n.as_name());
    let type_ref = TypeRef::from_ast_opt(node.ascribed_type());
    let sig = ConstData { name, type_ref };
    Arc::new(sig)
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
        db.static_data(self)
    }

    /// Builds a resolver for code inside this item.
    pub(crate) fn resolver(self, db: &impl HirDatabase) -> Resolver {
        // take the outer scope...
        self.module(db).resolver(db)
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
        self.trait_data(db).name().clone()
    }

    pub fn items(self, db: &impl DefDatabase) -> Vec<AssocItem> {
        self.trait_data(db).items().to_vec()
    }

    fn direct_super_traits(self, db: &impl HirDatabase) -> Vec<Trait> {
        let resolver = self.resolver(db);
        // returning the iterator directly doesn't easily work because of
        // lifetime problems, but since there usually shouldn't be more than a
        // few direct traits this should be fine (we could even use some kind of
        // SmallVec if performance is a concern)
        self.generic_params(db)
            .where_predicates
            .iter()
            .filter_map(|pred| match &pred.type_ref {
                TypeRef::Path(p) if p.as_ident() == Some(&name::SELF_TYPE) => pred.bound.as_path(),
                _ => None,
            })
            .filter_map(|path| match resolver.resolve_path_in_type_ns_fully(db, path) {
                Some(TypeNs::Trait(t)) => Some(t),
                _ => None,
            })
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
        trait_data
            .items()
            .iter()
            .filter_map(|item| match item {
                AssocItem::TypeAlias(t) => Some(*t),
                _ => None,
            })
            .find(|t| &t.name(db) == name)
    }

    pub fn associated_type_by_name_including_super_traits(
        self,
        db: &impl HirDatabase,
        name: &Name,
    ) -> Option<TypeAlias> {
        self.all_super_traits(db).into_iter().find_map(|t| t.associated_type_by_name(db, name))
    }

    pub(crate) fn trait_data(self, db: &impl DefDatabase) -> Arc<TraitData> {
        db.trait_data(self)
    }

    pub fn trait_ref(self, db: &impl HirDatabase) -> TraitRef {
        TraitRef::for_trait(db, self)
    }

    pub fn is_auto(self, db: &impl DefDatabase) -> bool {
        self.trait_data(db).is_auto()
    }

    pub(crate) fn resolver(self, db: &impl DefDatabase) -> Resolver {
        let r = self.module(db).resolver(db);
        // add generic params, if present
        let p = self.generic_params(db);
        let r = if !p.params.is_empty() { r.push_generic_params_scope(p) } else { r };
        r
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeAlias {
    pub(crate) id: TypeAliasId,
}

impl TypeAlias {
    pub fn module(self, db: &impl DefDatabase) -> Module {
        Module { id: self.id.module(db) }
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        Some(self.module(db).krate())
    }

    /// The containing impl block, if this is a method.
    pub fn impl_block(self, db: &impl DefDatabase) -> Option<ImplBlock> {
        let module_impls = db.impls_in_module(self.module(db));
        ImplBlock::containing(module_impls, self.into())
    }

    /// The containing trait, if this is a trait method definition.
    pub fn parent_trait(self, db: &impl DefDatabase) -> Option<Trait> {
        db.trait_items_index(self.module(db)).get_parent_trait(self.into())
    }

    pub fn container(self, db: &impl DefDatabase) -> Option<Container> {
        if let Some(impl_block) = self.impl_block(db) {
            Some(impl_block.into())
        } else if let Some(trait_) = self.parent_trait(db) {
            Some(trait_.into())
        } else {
            None
        }
    }

    pub fn type_ref(self, db: &impl DefDatabase) -> Option<TypeRef> {
        db.type_alias_data(self).type_ref.clone()
    }

    pub fn ty(self, db: &impl HirDatabase) -> Ty {
        db.type_for_def(self.into(), Namespace::Types)
    }

    pub fn name(self, db: &impl DefDatabase) -> Name {
        db.type_alias_data(self).name.clone()
    }

    /// Builds a resolver for the type references in this type alias.
    pub(crate) fn resolver(self, db: &impl HirDatabase) -> Resolver {
        // take the outer scope...
        let r = self
            .impl_block(db)
            .map(|ib| ib.resolver(db))
            .unwrap_or_else(|| self.module(db).resolver(db));
        // ...and add generic params, if present
        let p = self.generic_params(db);
        let r = if !p.params.is_empty() { r.push_generic_params_scope(p) } else { r };
        r
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

impl Container {
    pub(crate) fn resolver(self, db: &impl DefDatabase) -> Resolver {
        match self {
            Container::Trait(trait_) => trait_.resolver(db),
            Container::ImplBlock(impl_block) => impl_block.resolver(db),
        }
    }
}

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

impl From<AssocItem> for crate::generics::GenericDef {
    fn from(item: AssocItem) -> Self {
        match item {
            AssocItem::Function(f) => f.into(),
            AssocItem::Const(c) => c.into(),
            AssocItem::TypeAlias(t) => t.into(),
        }
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

    pub fn container(self, db: &impl DefDatabase) -> Container {
        match self {
            AssocItem::Function(f) => f.container(db),
            AssocItem::Const(c) => c.container(db),
            AssocItem::TypeAlias(t) => t.container(db),
        }
        .expect("AssocItem without container")
    }
}
