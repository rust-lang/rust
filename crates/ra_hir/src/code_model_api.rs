use std::sync::Arc;

use relative_path::RelativePathBuf;
use ra_db::{CrateId, FileId};
use ra_syntax::{ast::self, TreeArc, SyntaxNode};

use crate::{
    Name, ScopesWithSyntaxMapping, Ty, HirFileId,
    type_ref::TypeRef,
    nameres::{ModuleScope, lower::ImportId},
    HirDatabase, PersistentHirDatabase,
    expr::{Body, BodySyntaxMapping},
    ty::InferenceResult,
    adt::{EnumVariantId, StructFieldId, VariantDef},
    generics::GenericParams,
    docs::{Documentation, Docs, docs_from_ast},
    module_tree::ModuleId,
    ids::{FunctionId, StructId, EnumId, AstItemDef, ConstId, StaticId, TraitId, TypeId},
    impl_block::ImplId,
    resolve::Resolver,
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
    pub fn crate_id(&self) -> CrateId {
        self.crate_id
    }
    pub fn dependencies(&self, db: &impl PersistentHirDatabase) -> Vec<CrateDependency> {
        self.dependencies_impl(db)
    }
    pub fn root_module(&self, db: &impl PersistentHirDatabase) -> Option<Module> {
        self.root_module_impl(db)
    }
}

#[derive(Debug)]
pub enum Def {
    Item,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Module {
    pub(crate) krate: Crate,
    pub(crate) module_id: ModuleId,
}

/// The defs which can be visible in the module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleDef {
    Module(Module),
    Function(Function),
    Struct(Struct),
    Enum(Enum),
    // Can't be directly declared, but can be imported.
    EnumVariant(EnumVariant),
    Const(Const),
    Static(Static),
    Trait(Trait),
    Type(Type),
}
impl_froms!(ModuleDef: Module, Function, Struct, Enum, EnumVariant, Const, Static, Trait, Type);

pub enum ModuleSource {
    SourceFile(TreeArc<ast::SourceFile>),
    Module(TreeArc<ast::Module>),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Problem {
    UnresolvedModule { candidate: RelativePathBuf },
    NotDirOwner { move_to: RelativePathBuf, candidate: RelativePathBuf },
}

impl Module {
    /// Name of this module.
    pub fn name(&self, db: &impl HirDatabase) -> Option<Name> {
        self.name_impl(db)
    }

    /// Returns a node which defines this module. That is, a file or a `mod foo {}` with items.
    pub fn definition_source(&self, db: &impl PersistentHirDatabase) -> (FileId, ModuleSource) {
        self.definition_source_impl(db)
    }

    /// Returns a node which declares this module, either a `mod foo;` or a `mod foo {}`.
    /// `None` for the crate root.
    pub fn declaration_source(
        &self,
        db: &impl HirDatabase,
    ) -> Option<(FileId, TreeArc<ast::Module>)> {
        self.declaration_source_impl(db)
    }

    /// Returns the syntax of the last path segment corresponding to this import
    pub fn import_source(
        &self,
        db: &impl HirDatabase,
        import: ImportId,
    ) -> TreeArc<ast::PathSegment> {
        self.import_source_impl(db, import)
    }

    /// Returns the syntax of the impl block in this module
    pub fn impl_source(&self, db: &impl HirDatabase, impl_id: ImplId) -> TreeArc<ast::ImplBlock> {
        self.impl_source_impl(db, impl_id)
    }

    /// Returns the crate this module is part of.
    pub fn krate(&self, _db: &impl PersistentHirDatabase) -> Option<Crate> {
        Some(self.krate)
    }

    /// Topmost parent of this module. Every module has a `crate_root`, but some
    /// might be missing `krate`. This can happen if a module's file is not included
    /// in the module tree of any target in Cargo.toml.
    pub fn crate_root(&self, db: &impl PersistentHirDatabase) -> Module {
        self.crate_root_impl(db)
    }

    /// Finds a child module with the specified name.
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Option<Module> {
        self.child_impl(db, name)
    }

    /// Iterates over all child modules.
    pub fn children(&self, db: &impl PersistentHirDatabase) -> impl Iterator<Item = Module> {
        self.children_impl(db)
    }

    /// Finds a parent module.
    pub fn parent(&self, db: &impl PersistentHirDatabase) -> Option<Module> {
        self.parent_impl(db)
    }

    pub fn path_to_root(&self, db: &impl HirDatabase) -> Vec<Module> {
        let mut res = vec![self.clone()];
        let mut curr = self.clone();
        while let Some(next) = curr.parent(db) {
            res.push(next.clone());
            curr = next
        }
        res
    }

    /// Returns a `ModuleScope`: a set of items, visible in this module.
    pub fn scope(&self, db: &impl HirDatabase) -> ModuleScope {
        db.item_map(self.krate)[self.module_id].clone()
    }

    pub fn problems(&self, db: &impl HirDatabase) -> Vec<(TreeArc<SyntaxNode>, Problem)> {
        self.problems_impl(db)
    }

    pub fn resolver(&self, db: &impl HirDatabase) -> Resolver {
        let item_map = db.item_map(self.krate);
        Resolver::default().push_module_scope(item_map, *self)
    }
}

impl Docs for Module {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        self.declaration_source(db).and_then(|it| docs_from_ast(&*it.1))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructField {
    pub(crate) parent: VariantDef,
    pub(crate) id: StructFieldId,
}

#[derive(Debug)]
pub enum FieldSource {
    Named(TreeArc<ast::NamedFieldDef>),
    Pos(TreeArc<ast::PosFieldDef>),
}

impl StructField {
    pub fn name(&self, db: &impl HirDatabase) -> Name {
        self.parent.variant_data(db).fields().unwrap()[self.id].name.clone()
    }

    pub fn source(&self, db: &impl PersistentHirDatabase) -> (HirFileId, FieldSource) {
        self.source_impl(db)
    }

    pub fn ty(&self, db: &impl HirDatabase) -> Ty {
        db.type_for_field(*self)
    }

    pub fn parent_def(&self, _db: &impl HirDatabase) -> VariantDef {
        self.parent
    }
}

impl Docs for StructField {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        match self.source(db).1 {
            FieldSource::Named(named) => docs_from_ast(&*named),
            FieldSource::Pos(..) => return None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Struct {
    pub(crate) id: StructId,
}

impl Struct {
    pub fn source(&self, db: &impl PersistentHirDatabase) -> (HirFileId, TreeArc<ast::StructDef>) {
        self.id.source(db)
    }

    pub fn module(&self, db: &impl HirDatabase) -> Module {
        self.id.module(db)
    }

    pub fn name(&self, db: &impl HirDatabase) -> Option<Name> {
        db.struct_data(*self).name.clone()
    }

    pub fn fields(&self, db: &impl HirDatabase) -> Vec<StructField> {
        db.struct_data(*self)
            .variant_data
            .fields()
            .into_iter()
            .flat_map(|it| it.iter())
            .map(|(id, _)| StructField { parent: (*self).into(), id })
            .collect()
    }

    pub fn field(&self, db: &impl HirDatabase, name: &Name) -> Option<StructField> {
        db.struct_data(*self)
            .variant_data
            .fields()
            .into_iter()
            .flat_map(|it| it.iter())
            .find(|(_id, data)| data.name == *name)
            .map(|(id, _)| StructField { parent: (*self).into(), id })
    }

    pub fn generic_params(&self, db: &impl PersistentHirDatabase) -> Arc<GenericParams> {
        db.generic_params((*self).into())
    }

    pub fn ty(&self, db: &impl HirDatabase) -> Ty {
        db.type_for_def((*self).into())
    }

    // TODO move to a more general type
    /// Builds a resolver for type references inside this struct.
    pub fn resolver(&self, db: &impl HirDatabase) -> Resolver {
        // take the outer scope...
        let r = self.module(db).resolver(db);
        // ...and add generic params, if present
        let p = self.generic_params(db);
        let r = if !p.params.is_empty() { r.push_generic_params_scope(p) } else { r };
        r
    }
}

impl Docs for Struct {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Enum {
    pub(crate) id: EnumId,
}

impl Enum {
    pub fn source(&self, db: &impl PersistentHirDatabase) -> (HirFileId, TreeArc<ast::EnumDef>) {
        self.id.source(db)
    }

    pub fn module(&self, db: &impl HirDatabase) -> Module {
        self.id.module(db)
    }

    pub fn name(&self, db: &impl HirDatabase) -> Option<Name> {
        db.enum_data(*self).name.clone()
    }

    pub fn variants(&self, db: &impl HirDatabase) -> Vec<EnumVariant> {
        db.enum_data(*self)
            .variants
            .iter()
            .map(|(id, _)| EnumVariant { parent: *self, id })
            .collect()
    }

    pub fn variant(&self, db: &impl PersistentHirDatabase, name: &Name) -> Option<EnumVariant> {
        db.enum_data(*self)
            .variants
            .iter()
            .find(|(_id, data)| data.name.as_ref() == Some(name))
            .map(|(id, _)| EnumVariant { parent: *self, id })
    }

    pub fn generic_params(&self, db: &impl PersistentHirDatabase) -> Arc<GenericParams> {
        db.generic_params((*self).into())
    }

    pub fn ty(&self, db: &impl HirDatabase) -> Ty {
        db.type_for_def((*self).into())
    }

    // TODO move to a more general type
    /// Builds a resolver for type references inside this struct.
    pub fn resolver(&self, db: &impl HirDatabase) -> Resolver {
        // take the outer scope...
        let r = self.module(db).resolver(db);
        // ...and add generic params, if present
        let p = self.generic_params(db);
        let r = if !p.params.is_empty() { r.push_generic_params_scope(p) } else { r };
        r
    }
}

impl Docs for Enum {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumVariant {
    pub(crate) parent: Enum,
    pub(crate) id: EnumVariantId,
}

impl EnumVariant {
    pub fn source(
        &self,
        db: &impl PersistentHirDatabase,
    ) -> (HirFileId, TreeArc<ast::EnumVariant>) {
        self.source_impl(db)
    }
    pub fn module(&self, db: &impl HirDatabase) -> Module {
        self.parent.module(db)
    }
    pub fn parent_enum(&self, _db: &impl PersistentHirDatabase) -> Enum {
        self.parent
    }

    pub fn name(&self, db: &impl HirDatabase) -> Option<Name> {
        db.enum_data(self.parent).variants[self.id].name.clone()
    }

    pub fn fields(&self, db: &impl HirDatabase) -> Vec<StructField> {
        self.variant_data(db)
            .fields()
            .into_iter()
            .flat_map(|it| it.iter())
            .map(|(id, _)| StructField { parent: (*self).into(), id })
            .collect()
    }

    pub fn field(&self, db: &impl HirDatabase, name: &Name) -> Option<StructField> {
        self.variant_data(db)
            .fields()
            .into_iter()
            .flat_map(|it| it.iter())
            .find(|(_id, data)| data.name == *name)
            .map(|(id, _)| StructField { parent: (*self).into(), id })
    }
}

impl Docs for EnumVariant {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Function {
    pub(crate) id: FunctionId,
}

pub use crate::expr::ScopeEntryWithSyntax;

/// The declared signature of a function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnSignature {
    pub(crate) name: Name,
    pub(crate) params: Vec<TypeRef>,
    pub(crate) ret_type: TypeRef,
    /// True if the first param is `self`. This is relevant to decide whether this
    /// can be called as a method.
    pub(crate) has_self_param: bool,
}

impl FnSignature {
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
    pub fn source(&self, db: &impl PersistentHirDatabase) -> (HirFileId, TreeArc<ast::FnDef>) {
        self.id.source(db)
    }

    pub fn module(&self, db: &impl HirDatabase) -> Module {
        self.id.module(db)
    }

    pub fn name(&self, db: &impl HirDatabase) -> Name {
        self.signature(db).name.clone()
    }

    pub fn body_syntax_mapping(&self, db: &impl HirDatabase) -> Arc<BodySyntaxMapping> {
        db.body_syntax_mapping(*self)
    }

    pub fn body(&self, db: &impl HirDatabase) -> Arc<Body> {
        db.body_hir(*self)
    }

    pub fn scopes(&self, db: &impl HirDatabase) -> ScopesWithSyntaxMapping {
        let scopes = db.expr_scopes(*self);
        let syntax_mapping = db.body_syntax_mapping(*self);
        ScopesWithSyntaxMapping { scopes, syntax_mapping }
    }

    pub fn signature(&self, db: &impl HirDatabase) -> Arc<FnSignature> {
        db.fn_signature(*self)
    }

    pub fn infer(&self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(*self)
    }

    pub fn generic_params(&self, db: &impl PersistentHirDatabase) -> Arc<GenericParams> {
        db.generic_params((*self).into())
    }

    // TODO move to a more general type for 'body-having' items
    /// Builds a resolver for code inside this item.
    pub fn resolver(&self, db: &impl HirDatabase) -> Resolver {
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

impl Docs for Function {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Const {
    pub(crate) id: ConstId,
}

impl Const {
    pub fn source(&self, db: &impl PersistentHirDatabase) -> (HirFileId, TreeArc<ast::ConstDef>) {
        self.id.source(db)
    }
}

impl Docs for Const {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Static {
    pub(crate) id: StaticId,
}

impl Static {
    pub fn source(&self, db: &impl PersistentHirDatabase) -> (HirFileId, TreeArc<ast::StaticDef>) {
        self.id.source(db)
    }
}

impl Docs for Static {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Trait {
    pub(crate) id: TraitId,
}

impl Trait {
    pub fn source(&self, db: &impl PersistentHirDatabase) -> (HirFileId, TreeArc<ast::TraitDef>) {
        self.id.source(db)
    }

    pub fn generic_params(&self, db: &impl PersistentHirDatabase) -> Arc<GenericParams> {
        db.generic_params((*self).into())
    }
}

impl Docs for Trait {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Type {
    pub(crate) id: TypeId,
}

impl Type {
    pub fn source(&self, db: &impl PersistentHirDatabase) -> (HirFileId, TreeArc<ast::TypeDef>) {
        self.id.source(db)
    }

    pub fn generic_params(&self, db: &impl PersistentHirDatabase) -> Arc<GenericParams> {
        db.generic_params((*self).into())
    }
}

impl Docs for Type {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}
