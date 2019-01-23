use std::sync::Arc;

use relative_path::RelativePathBuf;
use ra_db::{CrateId, FileId};
use ra_syntax::{ast::self, TreeArc, SyntaxNode};

use crate::{
    Name, DefId, Path, PerNs, ScopesWithSyntaxMapping, Ty, HirFileId,
    type_ref::TypeRef,
    nameres::{ModuleScope, lower::ImportId},
    db::HirDatabase,
    expr::BodySyntaxMapping,
    ty::InferenceResult,
    adt::VariantData,
    generics::GenericParams,
    code_model_impl::def_id_to_ast,
    docs::{Documentation, Docs, docs_from_ast},
    module_tree::ModuleId,
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
    pub fn dependencies(&self, db: &impl HirDatabase) -> Vec<CrateDependency> {
        self.dependencies_impl(db)
    }
    pub fn root_module(&self, db: &impl HirDatabase) -> Option<Module> {
        self.root_module_impl(db)
    }
}

#[derive(Debug)]
pub enum Def {
    Struct(Struct),
    Enum(Enum),
    EnumVariant(EnumVariant),
    Function(Function),
    Const(Const),
    Static(Static),
    Trait(Trait),
    Type(Type),
    Item,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Module {
    pub(crate) krate: CrateId,
    pub(crate) module_id: ModuleId,
}

/// The defs which can be visible in the module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleDef {
    Module(Module),
    Def(DefId),
}

impl Into<ModuleDef> for Module {
    fn into(self) -> ModuleDef {
        ModuleDef::Module(self)
    }
}

impl Into<ModuleDef> for DefId {
    fn into(self) -> ModuleDef {
        ModuleDef::Def(self)
    }
}

pub enum ModuleSource {
    SourceFile(TreeArc<ast::SourceFile>),
    Module(TreeArc<ast::Module>),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Problem {
    UnresolvedModule {
        candidate: RelativePathBuf,
    },
    NotDirOwner {
        move_to: RelativePathBuf,
        candidate: RelativePathBuf,
    },
}

impl Module {
    /// Name of this module.
    pub fn name(&self, db: &impl HirDatabase) -> Option<Name> {
        self.name_impl(db)
    }

    /// Returns a node which defines this module. That is, a file or a `mod foo {}` with items.
    pub fn definition_source(&self, db: &impl HirDatabase) -> (FileId, ModuleSource) {
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

    /// Returns the crate this module is part of.
    pub fn krate(&self, db: &impl HirDatabase) -> Option<Crate> {
        self.krate_impl(db)
    }

    /// Topmost parent of this module. Every module has a `crate_root`, but some
    /// might be missing `krate`. This can happen if a module's file is not included
    /// in the module tree of any target in Cargo.toml.
    pub fn crate_root(&self, db: &impl HirDatabase) -> Module {
        self.crate_root_impl(db)
    }

    /// Finds a child module with the specified name.
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Option<Module> {
        self.child_impl(db, name)
    }

    /// Iterates over all child modules.
    pub fn children(&self, db: &impl HirDatabase) -> impl Iterator<Item = Module> {
        self.children_impl(db)
    }

    /// Finds a parent module.
    pub fn parent(&self, db: &impl HirDatabase) -> Option<Module> {
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
        self.scope_impl(db)
    }

    pub fn resolve_path(&self, db: &impl HirDatabase, path: &Path) -> PerNs<ModuleDef> {
        self.resolve_path_impl(db, path)
    }

    pub fn problems(&self, db: &impl HirDatabase) -> Vec<(TreeArc<SyntaxNode>, Problem)> {
        self.problems_impl(db)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructField {
    parent: DefId,
    name: Name,
}

impl StructField {
    pub fn name(&self) -> &Name {
        &self.name
    }

    pub fn ty(&self, db: &impl HirDatabase) -> Option<Ty> {
        db.type_for_field(self.parent, self.name.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Struct {
    pub(crate) def_id: DefId,
}

impl Struct {
    pub fn def_id(&self) -> DefId {
        self.def_id
    }

    pub fn name(&self, db: &impl HirDatabase) -> Option<Name> {
        db.struct_data(self.def_id).name.clone()
    }

    pub fn fields(&self, db: &impl HirDatabase) -> Vec<StructField> {
        db.struct_data(self.def_id)
            .variant_data
            .fields()
            .iter()
            .map(|it| StructField {
                parent: self.def_id,
                name: it.name.clone(),
            })
            .collect()
    }

    pub fn source(&self, db: &impl HirDatabase) -> (HirFileId, TreeArc<ast::StructDef>) {
        def_id_to_ast(db, self.def_id)
    }

    pub fn generic_params(&self, db: &impl HirDatabase) -> Arc<GenericParams> {
        db.generic_params(self.def_id)
    }
}

impl Docs for Struct {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Enum {
    pub(crate) def_id: DefId,
}

impl Enum {
    pub(crate) fn new(def_id: DefId) -> Self {
        Enum { def_id }
    }

    pub fn def_id(&self) -> DefId {
        self.def_id
    }

    pub fn name(&self, db: &impl HirDatabase) -> Option<Name> {
        db.enum_data(self.def_id).name.clone()
    }

    pub fn variants(&self, db: &impl HirDatabase) -> Vec<(Name, EnumVariant)> {
        db.enum_data(self.def_id).variants.clone()
    }

    pub fn source(&self, db: &impl HirDatabase) -> (HirFileId, TreeArc<ast::EnumDef>) {
        def_id_to_ast(db, self.def_id)
    }

    pub fn generic_params(&self, db: &impl HirDatabase) -> Arc<GenericParams> {
        db.generic_params(self.def_id)
    }
}

impl Docs for Enum {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumVariant {
    pub(crate) def_id: DefId,
}

impl EnumVariant {
    pub(crate) fn new(def_id: DefId) -> Self {
        EnumVariant { def_id }
    }

    pub fn def_id(&self) -> DefId {
        self.def_id
    }

    pub fn parent_enum(&self, db: &impl HirDatabase) -> Enum {
        db.enum_variant_data(self.def_id).parent_enum.clone()
    }

    pub fn name(&self, db: &impl HirDatabase) -> Option<Name> {
        db.enum_variant_data(self.def_id).name.clone()
    }

    pub fn variant_data(&self, db: &impl HirDatabase) -> Arc<VariantData> {
        db.enum_variant_data(self.def_id).variant_data.clone()
    }

    pub fn fields(&self, db: &impl HirDatabase) -> Vec<StructField> {
        self.variant_data(db)
            .fields()
            .iter()
            .map(|it| StructField {
                parent: self.def_id,
                name: it.name.clone(),
            })
            .collect()
    }

    pub fn source(&self, db: &impl HirDatabase) -> (HirFileId, TreeArc<ast::EnumVariant>) {
        def_id_to_ast(db, self.def_id)
    }
}

impl Docs for EnumVariant {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Function {
    pub(crate) def_id: DefId,
}

pub use crate::code_model_impl::function::ScopeEntryWithSyntax;

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
    pub fn def_id(&self) -> DefId {
        self.def_id
    }

    pub fn source(&self, db: &impl HirDatabase) -> (HirFileId, TreeArc<ast::FnDef>) {
        def_id_to_ast(db, self.def_id)
    }

    pub fn body_syntax_mapping(&self, db: &impl HirDatabase) -> Arc<BodySyntaxMapping> {
        db.body_syntax_mapping(self.def_id)
    }

    pub fn scopes(&self, db: &impl HirDatabase) -> ScopesWithSyntaxMapping {
        let scopes = db.fn_scopes(self.def_id);
        let syntax_mapping = db.body_syntax_mapping(self.def_id);
        ScopesWithSyntaxMapping {
            scopes,
            syntax_mapping,
        }
    }

    pub fn signature(&self, db: &impl HirDatabase) -> Arc<FnSignature> {
        db.fn_signature(self.def_id)
    }

    pub fn infer(&self, db: &impl HirDatabase) -> Arc<InferenceResult> {
        db.infer(self.def_id)
    }

    pub fn generic_params(&self, db: &impl HirDatabase) -> Arc<GenericParams> {
        db.generic_params(self.def_id)
    }
}

impl Docs for Function {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Const {
    pub(crate) def_id: DefId,
}

impl Const {
    pub(crate) fn new(def_id: DefId) -> Const {
        Const { def_id }
    }

    pub fn source(&self, db: &impl HirDatabase) -> (HirFileId, TreeArc<ast::ConstDef>) {
        def_id_to_ast(db, self.def_id)
    }
}

impl Docs for Const {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Static {
    pub(crate) def_id: DefId,
}

impl Static {
    pub(crate) fn new(def_id: DefId) -> Static {
        Static { def_id }
    }

    pub fn source(&self, db: &impl HirDatabase) -> (HirFileId, TreeArc<ast::StaticDef>) {
        def_id_to_ast(db, self.def_id)
    }
}

impl Docs for Static {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Trait {
    pub(crate) def_id: DefId,
}

impl Trait {
    pub(crate) fn new(def_id: DefId) -> Trait {
        Trait { def_id }
    }

    pub fn source(&self, db: &impl HirDatabase) -> (HirFileId, TreeArc<ast::TraitDef>) {
        def_id_to_ast(db, self.def_id)
    }

    pub fn generic_params(&self, db: &impl HirDatabase) -> Arc<GenericParams> {
        db.generic_params(self.def_id)
    }
}

impl Docs for Trait {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Type {
    pub(crate) def_id: DefId,
}

impl Type {
    pub(crate) fn new(def_id: DefId) -> Type {
        Type { def_id }
    }

    pub fn source(&self, db: &impl HirDatabase) -> (HirFileId, TreeArc<ast::TypeDef>) {
        def_id_to_ast(db, self.def_id)
    }

    pub fn generic_params(&self, db: &impl HirDatabase) -> Arc<GenericParams> {
        db.generic_params(self.def_id)
    }
}

impl Docs for Type {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}
