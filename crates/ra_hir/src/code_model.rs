pub(crate) mod src;
pub(crate) mod docs;

use std::sync::Arc;

use ra_db::{CrateId, Edition, FileId, SourceRootId};
use ra_syntax::ast::{self, NameOwner, TypeAscriptionOwner};

use crate::{
    adt::{EnumVariantId, StructFieldId, VariantDef},
    diagnostics::DiagnosticSink,
    expr::{validation::ExprValidator, Body, BodySourceMap},
    generics::HasGenericParams,
    ids::{
        AstItemDef, ConstId, EnumId, FunctionId, MacroDefId, StaticId, StructId, TraitId,
        TypeAliasId,
    },
    impl_block::ImplBlock,
    name::{
        BOOL, CHAR, F32, F64, I128, I16, I32, I64, I8, ISIZE, SELF_TYPE, STR, U128, U16, U32, U64,
        U8, USIZE,
    },
    nameres::{CrateModuleId, ImportId, ModuleScope, Namespace},
    resolve::Resolver,
    traits::{TraitData, TraitItem},
    ty::{
        primitive::{FloatBitness, FloatTy, IntBitness, IntTy, Signedness},
        InferenceResult, TraitRef,
    },
    type_ref::Mutability,
    type_ref::TypeRef,
    AsName, AstDatabase, AstId, DefDatabase, Either, HasSource, HirDatabase, Name, Ty,
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
        let module_id = db.crate_def_map(self).root();
        let module = Module { krate: self, module_id };
        Some(module)
    }

    pub fn edition(self, db: &impl DefDatabase) -> Edition {
        let crate_graph = db.crate_graph();
        crate_graph.edition(self.crate_id)
    }

    // FIXME: should this be in source_binder?
    pub fn source_root_crates(db: &impl DefDatabase, source_root: SourceRootId) -> Vec<Crate> {
        let crate_ids = db.source_root_crates(source_root);
        crate_ids.iter().map(|&crate_id| Crate { crate_id }).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Module {
    pub(crate) krate: Crate,
    pub(crate) module_id: CrateModuleId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    Char,
    Bool,
    Str,
    Int(IntTy),
    Float(FloatTy),
}

impl BuiltinType {
    #[rustfmt::skip]
    pub(crate) const ALL: &'static [(Name, BuiltinType)] = &[
        (CHAR, BuiltinType::Char),
        (BOOL, BuiltinType::Bool),
        (STR, BuiltinType::Str),

        (ISIZE, BuiltinType::Int(IntTy { signedness: Signedness::Signed, bitness: IntBitness::Xsize })),
        (I8,    BuiltinType::Int(IntTy { signedness: Signedness::Signed, bitness: IntBitness::X8 })),
        (I16,   BuiltinType::Int(IntTy { signedness: Signedness::Signed, bitness: IntBitness::X16 })),
        (I32,   BuiltinType::Int(IntTy { signedness: Signedness::Signed, bitness: IntBitness::X32 })),
        (I64,   BuiltinType::Int(IntTy { signedness: Signedness::Signed, bitness: IntBitness::X64 })),
        (I128,  BuiltinType::Int(IntTy { signedness: Signedness::Signed, bitness: IntBitness::X128 })),

        (USIZE, BuiltinType::Int(IntTy { signedness: Signedness::Unsigned, bitness: IntBitness::Xsize })),
        (U8,    BuiltinType::Int(IntTy { signedness: Signedness::Unsigned, bitness: IntBitness::X8 })),
        (U16,   BuiltinType::Int(IntTy { signedness: Signedness::Unsigned, bitness: IntBitness::X16 })),
        (U32,   BuiltinType::Int(IntTy { signedness: Signedness::Unsigned, bitness: IntBitness::X32 })),
        (U64,   BuiltinType::Int(IntTy { signedness: Signedness::Unsigned, bitness: IntBitness::X64 })),
        (U128,  BuiltinType::Int(IntTy { signedness: Signedness::Unsigned, bitness: IntBitness::X128 })),

        (F32, BuiltinType::Float(FloatTy { bitness: FloatBitness::X32 })),
        (F64, BuiltinType::Float(FloatTy { bitness: FloatBitness::X64 })),
    ];
}

/// The defs which can be visible in the module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModuleDef {
    Module(Module),
    Function(Function),
    Struct(Struct),
    Union(Union),
    Enum(Enum),
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
    Struct,
    Union,
    Enum,
    EnumVariant,
    Const,
    Static,
    Trait,
    TypeAlias,
    BuiltinType
);

pub enum ModuleSource {
    SourceFile(ast::SourceFile),
    Module(ast::Module),
}

impl ModuleSource {
    pub(crate) fn new(
        db: &(impl DefDatabase + AstDatabase),
        file_id: Option<FileId>,
        decl_id: Option<AstId<ast::Module>>,
    ) -> ModuleSource {
        match (file_id, decl_id) {
            (Some(file_id), _) => {
                let source_file = db.parse(file_id).tree().to_owned();
                ModuleSource::SourceFile(source_file)
            }
            (None, Some(item_id)) => {
                let module = item_id.to_node(db);
                assert!(module.item_list().is_some(), "expected inline module");
                ModuleSource::Module(module)
            }
            (None, None) => panic!(),
        }
    }
}

impl Module {
    /// Name of this module.
    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        let def_map = db.crate_def_map(self.krate);
        let parent = def_map[self.module_id].parent?;
        def_map[parent].children.iter().find_map(|(name, module_id)| {
            if *module_id == self.module_id {
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
    pub fn krate(self, _db: &impl DefDatabase) -> Option<Crate> {
        Some(self.krate)
    }

    /// Topmost parent of this module. Every module has a `crate_root`, but some
    /// might be missing `krate`. This can happen if a module's file is not included
    /// in the module tree of any target in `Cargo.toml`.
    pub fn crate_root(self, db: &impl DefDatabase) -> Module {
        let def_map = db.crate_def_map(self.krate);
        self.with_module_id(def_map.root())
    }

    /// Finds a child module with the specified name.
    pub fn child(self, db: &impl HirDatabase, name: &Name) -> Option<Module> {
        let def_map = db.crate_def_map(self.krate);
        let child_id = def_map[self.module_id].children.get(name)?;
        Some(self.with_module_id(*child_id))
    }

    /// Iterates over all child modules.
    pub fn children(self, db: &impl DefDatabase) -> impl Iterator<Item = Module> {
        let def_map = db.crate_def_map(self.krate);
        let children = def_map[self.module_id]
            .children
            .iter()
            .map(|(_, module_id)| self.with_module_id(*module_id))
            .collect::<Vec<_>>();
        children.into_iter()
    }

    /// Finds a parent module.
    pub fn parent(self, db: &impl DefDatabase) -> Option<Module> {
        let def_map = db.crate_def_map(self.krate);
        let parent_id = def_map[self.module_id].parent?;
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
    pub fn scope(self, db: &impl HirDatabase) -> ModuleScope {
        db.crate_def_map(self.krate)[self.module_id].scope.clone()
    }

    pub fn diagnostics(self, db: &impl HirDatabase, sink: &mut DiagnosticSink) {
        db.crate_def_map(self.krate).add_diagnostics(db, self.module_id, sink);
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
                if let crate::ImplItem::Method(f) = item {
                    f.diagnostics(db, sink);
                }
            }
        }
    }

    pub(crate) fn resolver(self, db: &impl DefDatabase) -> Resolver {
        let def_map = db.crate_def_map(self.krate);
        Resolver::default().push_module_scope(def_map, self.module_id)
    }

    pub fn declarations(self, db: &impl DefDatabase) -> Vec<ModuleDef> {
        let def_map = db.crate_def_map(self.krate);
        def_map[self.module_id]
            .scope
            .entries()
            .filter_map(|(_name, res)| if res.import.is_none() { Some(res.def) } else { None })
            .flat_map(|per_ns| {
                per_ns.take_types().into_iter().chain(per_ns.take_values().into_iter())
            })
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
        Module { module_id, krate: self.krate }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructField {
    pub(crate) parent: VariantDef,
    pub(crate) id: StructFieldId,
}

#[derive(Debug)]
pub enum FieldSource {
    Named(ast::NamedFieldDef),
    Pos(ast::PosFieldDef),
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
    pub fn module(self, db: &impl HirDatabase) -> Module {
        self.id.module(db)
    }

    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        db.struct_data(self).name.clone()
    }

    pub fn fields(self, db: &impl HirDatabase) -> Vec<StructField> {
        db.struct_data(self)
            .variant_data
            .fields()
            .into_iter()
            .flat_map(|it| it.iter())
            .map(|(id, _)| StructField { parent: self.into(), id })
            .collect()
    }

    pub fn field(self, db: &impl HirDatabase, name: &Name) -> Option<StructField> {
        db.struct_data(self)
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
    pub(crate) id: StructId,
}

impl Union {
    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        db.struct_data(Struct { id: self.id }).name.clone()
    }

    pub fn module(self, db: &impl HirDatabase) -> Module {
        self.id.module(db)
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
    pub fn module(self, db: &impl HirDatabase) -> Module {
        self.id.module(db)
    }

    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        db.enum_data(self).name.clone()
    }

    pub fn variants(self, db: &impl DefDatabase) -> Vec<EnumVariant> {
        db.enum_data(self).variants.iter().map(|(id, _)| EnumVariant { parent: self, id }).collect()
    }

    pub fn variant(self, db: &impl DefDatabase, name: &Name) -> Option<EnumVariant> {
        db.enum_data(self)
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
        r
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumVariant {
    pub(crate) parent: Enum,
    pub(crate) id: EnumVariantId,
}

impl EnumVariant {
    pub fn module(self, db: &impl HirDatabase) -> Module {
        self.parent.module(db)
    }
    pub fn parent_enum(self, _db: &impl DefDatabase) -> Enum {
        self.parent
    }

    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        db.enum_data(self.parent).variants[self.id].name.clone()
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
    pub fn infer(self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(self)
    }

    pub fn body(self, db: &impl HirDatabase) -> Arc<Body> {
        db.body_hir(self)
    }

    pub fn body_source_map(self, db: &impl HirDatabase) -> Arc<BodySourceMap> {
        db.body_with_source_map(self).1
    }

    /// Builds a resolver for code inside this item.
    pub(crate) fn resolver(self, db: &impl HirDatabase) -> Resolver {
        match self {
            DefWithBody::Const(c) => c.resolver(db),
            DefWithBody::Function(f) => f.resolver(db),
            DefWithBody::Static(s) => s.resolver(db),
        }
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
                    let self_type = TypeRef::Path(SELF_TYPE.into());
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
        self.id.module(db)
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
        self.id.module(db)
    }

    pub fn data(self, db: &impl HirDatabase) -> Arc<ConstData> {
        db.const_data(self)
    }

    pub fn infer(self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(self.into())
    }

    /// The containing impl block, if this is a method.
    pub fn impl_block(self, db: &impl DefDatabase) -> Option<ImplBlock> {
        let module_impls = db.impls_in_module(self.module(db));
        ImplBlock::containing(module_impls, self.into())
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
    pub(crate) name: Name,
    pub(crate) type_ref: TypeRef,
}

impl ConstData {
    pub fn name(&self) -> &Name {
        &self.name
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
    let name = node.name().map(|n| n.as_name()).unwrap_or_else(Name::missing);
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
        self.id.module(db)
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
        self.id.module(db)
    }

    pub fn name(self, db: &impl DefDatabase) -> Option<Name> {
        self.trait_data(db).name().clone()
    }

    pub fn items(self, db: &impl DefDatabase) -> Vec<TraitItem> {
        self.trait_data(db).items().to_vec()
    }

    pub fn associated_type_by_name(self, db: &impl DefDatabase, name: Name) -> Option<TypeAlias> {
        let trait_data = self.trait_data(db);
        trait_data
            .items()
            .iter()
            .filter_map(|item| match item {
                TraitItem::TypeAlias(t) => Some(*t),
                _ => None,
            })
            .find(|t| t.name(db) == name)
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
        self.id.module(db)
    }

    pub fn krate(self, db: &impl DefDatabase) -> Option<Crate> {
        self.module(db).krate(db)
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
