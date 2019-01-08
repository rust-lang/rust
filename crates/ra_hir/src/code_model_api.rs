use std::sync::Arc;

use relative_path::RelativePathBuf;
use ra_db::{CrateId, Cancelable, FileId};
use ra_syntax::{ast, TreePtr, SyntaxNode};

use crate::{
    Name, DefId, Path, PerNs, ScopesWithSyntaxMapping, Ty,
    type_ref::TypeRef,
    nameres::ModuleScope,
    db::HirDatabase,
    expr::BodySyntaxMapping,
    ty::InferenceResult,
    adt::VariantData,
};

/// hir::Crate describes a single crate. It's the main interface with which
/// a crate's dependencies interact. Mostly, it should be just a proxy for the
/// root module.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    pub fn dependencies(&self, db: &impl HirDatabase) -> Cancelable<Vec<CrateDependency>> {
        Ok(self.dependencies_impl(db))
    }
    pub fn root_module(&self, db: &impl HirDatabase) -> Cancelable<Option<Module>> {
        self.root_module_impl(db)
    }
}

pub enum Def {
    Module(Module),
    Struct(Struct),
    Enum(Enum),
    EnumVariant(EnumVariant),
    Function(Function),
    Item,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Module {
    pub(crate) def_id: DefId,
}

pub enum ModuleSource {
    SourceFile(TreePtr<ast::SourceFile>),
    Module(TreePtr<ast::Module>),
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
    pub fn name(&self, db: &impl HirDatabase) -> Cancelable<Option<Name>> {
        self.name_impl(db)
    }

    /// Returns a node which defines this module. That is, a file or a `mod foo {}` with items.
    pub fn definition_source(&self, db: &impl HirDatabase) -> Cancelable<(FileId, ModuleSource)> {
        self.definition_source_impl(db)
    }

    /// Returns a node which declares this module, either a `mod foo;` or a `mod foo {}`.
    /// `None` for the crate root.
    pub fn declaration_source(
        &self,
        db: &impl HirDatabase,
    ) -> Cancelable<Option<(FileId, TreePtr<ast::Module>)>> {
        self.declaration_source_impl(db)
    }

    /// Returns the crate this module is part of.
    pub fn krate(&self, db: &impl HirDatabase) -> Cancelable<Option<Crate>> {
        self.krate_impl(db)
    }

    /// Topmost parent of this module. Every module has a `crate_root`, but some
    /// might be missing `krate`. This can happen if a module's file is not included
    /// in the module tree of any target in Cargo.toml.
    pub fn crate_root(&self, db: &impl HirDatabase) -> Cancelable<Module> {
        self.crate_root_impl(db)
    }

    /// Finds a child module with the specified name.
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
    }

    /// Finds a parent module.
    pub fn parent(&self, db: &impl HirDatabase) -> Cancelable<Option<Module>> {
        self.parent_impl(db)
    }

    pub fn path_to_root(&self, db: &impl HirDatabase) -> Cancelable<Vec<Module>> {
        let mut res = vec![self.clone()];
        let mut curr = self.clone();
        while let Some(next) = curr.parent(db)? {
            res.push(next.clone());
            curr = next
        }
        Ok(res)
    }

    /// Returns a `ModuleScope`: a set of items, visible in this module.
    pub fn scope(&self, db: &impl HirDatabase) -> Cancelable<ModuleScope> {
        self.scope_impl(db)
    }

    pub fn resolve_path(&self, db: &impl HirDatabase, path: &Path) -> Cancelable<PerNs<DefId>> {
        self.resolve_path_impl(db, path)
    }

    pub fn problems(
        &self,
        db: &impl HirDatabase,
    ) -> Cancelable<Vec<(TreePtr<SyntaxNode>, Problem)>> {
        self.problems_impl(db)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructField {
    struct_: Struct,
    name: Name,
}

impl StructField {
    pub fn name(&self) -> &Name {
        &self.name
    }
    pub fn ty(&self, db: &impl HirDatabase) -> Cancelable<Option<Ty>> {
        db.type_for_field(self.struct_.def_id, self.name.clone())
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

    pub fn name(&self, db: &impl HirDatabase) -> Cancelable<Option<Name>> {
        Ok(db.struct_data(self.def_id)?.name.clone())
    }

    pub fn fields(&self, db: &impl HirDatabase) -> Cancelable<Vec<StructField>> {
        let res = db
            .struct_data(self.def_id)?
            .variant_data
            .fields()
            .iter()
            .map(|it| StructField {
                struct_: self.clone(),
                name: it.name.clone(),
            })
            .collect();
        Ok(res)
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

    pub fn name(&self, db: &impl HirDatabase) -> Cancelable<Option<Name>> {
        Ok(db.enum_data(self.def_id)?.name.clone())
    }

    pub fn variants(&self, db: &impl HirDatabase) -> Cancelable<Option<Vec<EnumVariant>>> {
        Ok(db.enum_data(self.def_id)?.variants.clone())
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

    pub fn parent_enum(&self, db: &impl HirDatabase) -> Cancelable<Enum> {
        Ok(db.enum_variant_data(self.def_id)?.parent_enum.clone())
    }

    pub fn name(&self, db: &impl HirDatabase) -> Cancelable<Option<Name>> {
        Ok(db.enum_variant_data(self.def_id)?.name.clone())
    }

    pub fn variant_data(&self, db: &impl HirDatabase) -> Cancelable<Arc<VariantData>> {
        Ok(db.enum_variant_data(self.def_id)?.variant_data.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Function {
    pub(crate) def_id: DefId,
}

/// The declared signature of a function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnSignature {
    pub(crate) args: Vec<TypeRef>,
    pub(crate) ret_type: TypeRef,
}

impl FnSignature {
    pub fn args(&self) -> &[TypeRef] {
        &self.args
    }

    pub fn ret_type(&self) -> &TypeRef {
        &self.ret_type
    }
}

impl Function {
    pub fn def_id(&self) -> DefId {
        self.def_id
    }

    pub fn source(&self, db: &impl HirDatabase) -> TreePtr<ast::FnDef> {
        self.source_impl(db)
    }

    pub fn body_syntax_mapping(&self, db: &impl HirDatabase) -> Cancelable<Arc<BodySyntaxMapping>> {
        db.body_syntax_mapping(self.def_id)
    }

    pub fn scopes(&self, db: &impl HirDatabase) -> Cancelable<ScopesWithSyntaxMapping> {
        let scopes = db.fn_scopes(self.def_id)?;
        let syntax_mapping = db.body_syntax_mapping(self.def_id)?;
        Ok(ScopesWithSyntaxMapping {
            scopes,
            syntax_mapping,
        })
    }

    pub fn signature(&self, db: &impl HirDatabase) -> Arc<FnSignature> {
        db.fn_signature(self.def_id)
    }

    pub fn infer(&self, db: &impl HirDatabase) -> Cancelable<Arc<InferenceResult>> {
        db.infer(self.def_id)
    }
}
