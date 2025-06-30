//! HIR (previously known as descriptors) provides a high-level object-oriented
//! access to Rust code.
//!
//! The principal difference between HIR and syntax trees is that HIR is bound
//! to a particular crate instance. That is, it has cfg flags and features
//! applied. So, the relation between syntax and HIR is many-to-one.
//!
//! HIR is the public API of the all of the compiler logic above syntax trees.
//! It is written in "OO" style. Each type is self contained (as in, it knows its
//! parents and full context). It should be "clean code".
//!
//! `hir_*` crates are the implementation of the compiler logic.
//! They are written in "ECS" style, with relatively little abstractions.
//! Many types are not self-contained, and explicitly use local indexes, arenas, etc.
//!
//! `hir` is what insulates the "we don't know how to actually write an incremental compiler"
//! from the ide with completions, hovers, etc. It is a (soft, internal) boundary:
//! <https://www.tedinski.com/2018/02/06/system-boundaries.html>.

#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]
#![recursion_limit = "512"]

mod attrs;
mod from_id;
mod has_source;
mod semantics;
mod source_analyzer;

pub mod db;
pub mod diagnostics;
pub mod symbols;
pub mod term_search;

mod display;

use std::{
    mem::discriminant,
    ops::{ControlFlow, Not},
};

use arrayvec::ArrayVec;
use base_db::{CrateDisplayName, CrateOrigin, LangCrateOrigin};
use either::Either;
use hir_def::{
    AdtId, AssocItemId, AssocItemLoc, AttrDefId, CallableDefId, ConstId, ConstParamId,
    CrateRootModuleId, DefWithBodyId, EnumId, EnumVariantId, ExternBlockId, ExternCrateId,
    FunctionId, GenericDefId, GenericParamId, HasModule, ImplId, ItemContainerId, LifetimeParamId,
    LocalFieldId, Lookup, MacroExpander, MacroId, ModuleId, StaticId, StructId, SyntheticSyntax,
    TraitAliasId, TupleId, TypeAliasId, TypeOrConstParamId, TypeParamId, UnionId,
    expr_store::{ExpressionStoreDiagnostics, ExpressionStoreSourceMap},
    hir::{
        BindingAnnotation, BindingId, Expr, ExprId, ExprOrPatId, LabelId, Pat,
        generics::{LifetimeParamData, TypeOrConstParamData, TypeParamProvenance},
    },
    item_tree::ImportAlias,
    layout::{self, ReprOptions, TargetDataLayout},
    nameres::{self, assoc::TraitItems, diagnostics::DefDiagnostic},
    per_ns::PerNs,
    resolver::{HasResolver, Resolver},
    signatures::{ImplFlags, StaticFlags, TraitFlags, VariantFields},
    src::HasSource as _,
    visibility::visibility_from_ast,
};
use hir_expand::{
    AstId, MacroCallKind, RenderedExpandError, ValueResult, attrs::collect_attrs,
    proc_macro::ProcMacroKind,
};
use hir_ty::{
    AliasTy, CallableSig, Canonical, CanonicalVarKinds, Cast, ClosureId, GenericArg,
    GenericArgData, Interner, ParamKind, QuantifiedWhereClause, Scalar, Substitution,
    TraitEnvironment, TraitRefExt, Ty, TyBuilder, TyDefId, TyExt, TyKind, TyLoweringDiagnostic,
    ValueTyDefId, WhereClause, all_super_traits, autoderef, check_orphan_rules,
    consteval::{ConstExt, try_const_usize, unknown_const_as_generic},
    diagnostics::BodyValidationDiagnostic,
    direct_super_traits, error_lifetime, known_const_to_ast,
    layout::{Layout as TyLayout, RustcEnumVariantIdx, RustcFieldIdx, TagEncoding},
    method_resolution,
    mir::{MutBorrowKind, interpret_mir},
    primitive::UintTy,
    traits::FnTrait,
};
use itertools::Itertools;
use nameres::diagnostics::DefDiagnosticKind;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use span::{AstIdNode, Edition, FileId};
use stdx::{format_to, impl_from, never, variance::PhantomCovariantLifetime};
use syntax::{
    AstNode, AstPtr, SmolStr, SyntaxNode, SyntaxNodePtr, T, TextRange, ToSmolStr,
    ast::{self, HasAttrs as _, HasName, HasVisibility as _},
    format_smolstr,
};
use triomphe::{Arc, ThinArc};

use crate::db::{DefDatabase, HirDatabase};

pub use crate::{
    attrs::{HasAttrs, resolve_doc_path_on},
    diagnostics::*,
    has_source::HasSource,
    semantics::{
        PathResolution, PathResolutionPerNs, Semantics, SemanticsImpl, SemanticsScope, TypeInfo,
        VisibleTraits,
    },
};

// Be careful with these re-exports.
//
// `hir` is the boundary between the compiler and the IDE. It should try hard to
// isolate the compiler from the ide, to allow the two to be refactored
// independently. Re-exporting something from the compiler is the sure way to
// breach the boundary.
//
// Generally, a refactoring which *removes* a name from this list is a good
// idea!
pub use {
    cfg::{CfgAtom, CfgExpr, CfgOptions},
    hir_def::{
        Complete,
        ImportPathConfig,
        attr::{AttrSourceMap, Attrs, AttrsWithOwner},
        find_path::PrefixKind,
        import_map,
        lang_item::LangItem,
        nameres::{DefMap, ModuleSource, crate_def_map},
        per_ns::Namespace,
        type_ref::{Mutability, TypeRef},
        visibility::Visibility,
        // FIXME: This is here since some queries take it as input that are used
        // outside of hir.
        {ModuleDefId, TraitId},
    },
    hir_expand::{
        EditionedFileId, ExpandResult, HirFileId, MacroCallId, MacroKind,
        attrs::{Attr, AttrId},
        change::ChangeWithProcMacros,
        files::{
            FilePosition, FilePositionWrapper, FileRange, FileRangeWrapper, HirFilePosition,
            HirFileRange, InFile, InFileWrapper, InMacroFile, InRealFile, MacroFilePosition,
            MacroFileRange,
        },
        inert_attr_macro::AttributeTemplate,
        mod_path::{ModPath, PathKind, tool_path},
        name::Name,
        prettify_macro_expansion,
        proc_macro::{ProcMacros, ProcMacrosBuilder},
        tt,
    },
    hir_ty::{
        CastError, DropGlue, FnAbi, PointerCast, Safety, Variance,
        consteval::ConstEvalError,
        diagnostics::UnsafetyReason,
        display::{ClosureStyle, DisplayTarget, HirDisplay, HirDisplayError, HirWrite},
        dyn_compatibility::{DynCompatibilityViolation, MethodViolationCode},
        layout::LayoutError,
        method_resolution::TyFingerprint,
        mir::{MirEvalError, MirLowerError},
    },
    // FIXME: Properly encapsulate mir
    hir_ty::{Interner as ChalkTyInterner, mir},
    intern::{Symbol, sym},
};

// These are negative re-exports: pub using these names is forbidden, they
// should remain private to hir internals.
#[allow(unused)]
use {
    hir_def::expr_store::path::Path,
    hir_expand::{
        name::AsName,
        span_map::{ExpansionSpanMap, RealSpanMap, SpanMap, SpanMapRef},
    },
};

/// hir::Crate describes a single crate. It's the main interface with which
/// a crate's dependencies interact. Mostly, it should be just a proxy for the
/// root module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Crate {
    pub(crate) id: base_db::Crate,
}

#[derive(Debug)]
pub struct CrateDependency {
    pub krate: Crate,
    pub name: Name,
}

impl Crate {
    pub fn origin(self, db: &dyn HirDatabase) -> CrateOrigin {
        self.id.data(db).origin.clone()
    }

    pub fn is_builtin(self, db: &dyn HirDatabase) -> bool {
        matches!(self.origin(db), CrateOrigin::Lang(_))
    }

    pub fn dependencies(self, db: &dyn HirDatabase) -> Vec<CrateDependency> {
        self.id
            .data(db)
            .dependencies
            .iter()
            .map(|dep| {
                let krate = Crate { id: dep.crate_id };
                let name = dep.as_name();
                CrateDependency { krate, name }
            })
            .collect()
    }

    pub fn reverse_dependencies(self, db: &dyn HirDatabase) -> Vec<Crate> {
        let all_crates = db.all_crates();
        all_crates
            .iter()
            .copied()
            .filter(|&krate| krate.data(db).dependencies.iter().any(|it| it.crate_id == self.id))
            .map(|id| Crate { id })
            .collect()
    }

    pub fn transitive_reverse_dependencies(
        self,
        db: &dyn HirDatabase,
    ) -> impl Iterator<Item = Crate> {
        db.transitive_rev_deps(self.id).into_iter().map(|id| Crate { id })
    }

    pub fn root_module(self) -> Module {
        Module { id: CrateRootModuleId::from(self.id).into() }
    }

    pub fn modules(self, db: &dyn HirDatabase) -> Vec<Module> {
        let def_map = crate_def_map(db, self.id);
        def_map.modules().map(|(id, _)| def_map.module_id(id).into()).collect()
    }

    pub fn root_file(self, db: &dyn HirDatabase) -> FileId {
        self.id.data(db).root_file_id
    }

    pub fn edition(self, db: &dyn HirDatabase) -> Edition {
        self.id.data(db).edition
    }

    pub fn version(self, db: &dyn HirDatabase) -> Option<String> {
        self.id.extra_data(db).version.clone()
    }

    pub fn display_name(self, db: &dyn HirDatabase) -> Option<CrateDisplayName> {
        self.id.extra_data(db).display_name.clone()
    }

    pub fn query_external_importables(
        self,
        db: &dyn DefDatabase,
        query: import_map::Query,
    ) -> impl Iterator<Item = (Either<ModuleDef, Macro>, Complete)> {
        let _p = tracing::info_span!("query_external_importables").entered();
        import_map::search_dependencies(db, self.into(), &query).into_iter().map(
            |(item, do_not_complete)| {
                let item = match ItemInNs::from(item) {
                    ItemInNs::Types(mod_id) | ItemInNs::Values(mod_id) => Either::Left(mod_id),
                    ItemInNs::Macros(mac_id) => Either::Right(mac_id),
                };
                (item, do_not_complete)
            },
        )
    }

    pub fn all(db: &dyn HirDatabase) -> Vec<Crate> {
        db.all_crates().iter().map(|&id| Crate { id }).collect()
    }

    /// Try to get the root URL of the documentation of a crate.
    pub fn get_html_root_url(self: &Crate, db: &dyn HirDatabase) -> Option<String> {
        // Look for #![doc(html_root_url = "...")]
        let attrs = db.attrs(AttrDefId::ModuleId(self.root_module().into()));
        let doc_url = attrs.by_key(sym::doc).find_string_value_in_tt(sym::html_root_url);
        doc_url.map(|s| s.trim_matches('"').trim_end_matches('/').to_owned() + "/")
    }

    pub fn cfg<'db>(&self, db: &'db dyn HirDatabase) -> &'db CfgOptions {
        self.id.cfg_options(db)
    }

    pub fn potential_cfg<'db>(&self, db: &'db dyn HirDatabase) -> &'db CfgOptions {
        let data = self.id.extra_data(db);
        data.potential_cfg_options.as_ref().unwrap_or_else(|| self.id.cfg_options(db))
    }

    pub fn to_display_target(self, db: &dyn HirDatabase) -> DisplayTarget {
        DisplayTarget::from_crate(db, self.id)
    }

    fn core(db: &dyn HirDatabase) -> Option<Crate> {
        db.all_crates()
            .iter()
            .copied()
            .find(|&krate| {
                matches!(krate.data(db).origin, CrateOrigin::Lang(LangCrateOrigin::Core))
            })
            .map(Crate::from)
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
    // FIXME: Rename to `EnumVariant`
    Variant(Variant),
    Const(Const),
    Static(Static),
    Trait(Trait),
    TraitAlias(TraitAlias),
    TypeAlias(TypeAlias),
    BuiltinType(BuiltinType),
    Macro(Macro),
}
impl_from!(
    Module,
    Function,
    Adt(Struct, Enum, Union),
    Variant,
    Const,
    Static,
    Trait,
    TraitAlias,
    TypeAlias,
    BuiltinType,
    Macro
    for ModuleDef
);

impl From<VariantDef> for ModuleDef {
    fn from(var: VariantDef) -> Self {
        match var {
            VariantDef::Struct(t) => Adt::from(t).into(),
            VariantDef::Union(t) => Adt::from(t).into(),
            VariantDef::Variant(t) => t.into(),
        }
    }
}

impl ModuleDef {
    pub fn module(self, db: &dyn HirDatabase) -> Option<Module> {
        match self {
            ModuleDef::Module(it) => it.parent(db),
            ModuleDef::Function(it) => Some(it.module(db)),
            ModuleDef::Adt(it) => Some(it.module(db)),
            ModuleDef::Variant(it) => Some(it.module(db)),
            ModuleDef::Const(it) => Some(it.module(db)),
            ModuleDef::Static(it) => Some(it.module(db)),
            ModuleDef::Trait(it) => Some(it.module(db)),
            ModuleDef::TraitAlias(it) => Some(it.module(db)),
            ModuleDef::TypeAlias(it) => Some(it.module(db)),
            ModuleDef::Macro(it) => Some(it.module(db)),
            ModuleDef::BuiltinType(_) => None,
        }
    }

    pub fn canonical_path(&self, db: &dyn HirDatabase, edition: Edition) -> Option<String> {
        let mut segments = vec![self.name(db)?];
        for m in self.module(db)?.path_to_root(db) {
            segments.extend(m.name(db))
        }
        segments.reverse();
        Some(segments.iter().map(|it| it.display(db, edition)).join("::"))
    }

    pub fn canonical_module_path(
        &self,
        db: &dyn HirDatabase,
    ) -> Option<impl Iterator<Item = Module>> {
        self.module(db).map(|it| it.path_to_root(db).into_iter().rev())
    }

    pub fn name(self, db: &dyn HirDatabase) -> Option<Name> {
        let name = match self {
            ModuleDef::Module(it) => it.name(db)?,
            ModuleDef::Const(it) => it.name(db)?,
            ModuleDef::Adt(it) => it.name(db),
            ModuleDef::Trait(it) => it.name(db),
            ModuleDef::TraitAlias(it) => it.name(db),
            ModuleDef::Function(it) => it.name(db),
            ModuleDef::Variant(it) => it.name(db),
            ModuleDef::TypeAlias(it) => it.name(db),
            ModuleDef::Static(it) => it.name(db),
            ModuleDef::Macro(it) => it.name(db),
            ModuleDef::BuiltinType(it) => it.name(),
        };
        Some(name)
    }

    pub fn diagnostics<'db>(
        self,
        db: &'db dyn HirDatabase,
        style_lints: bool,
    ) -> Vec<AnyDiagnostic<'db>> {
        let id = match self {
            ModuleDef::Adt(it) => match it {
                Adt::Struct(it) => it.id.into(),
                Adt::Enum(it) => it.id.into(),
                Adt::Union(it) => it.id.into(),
            },
            ModuleDef::Trait(it) => it.id.into(),
            ModuleDef::TraitAlias(it) => it.id.into(),
            ModuleDef::Function(it) => it.id.into(),
            ModuleDef::TypeAlias(it) => it.id.into(),
            ModuleDef::Module(it) => it.id.into(),
            ModuleDef::Const(it) => it.id.into(),
            ModuleDef::Static(it) => it.id.into(),
            ModuleDef::Variant(it) => it.id.into(),
            ModuleDef::BuiltinType(_) | ModuleDef::Macro(_) => return Vec::new(),
        };

        let mut acc = Vec::new();

        match self.as_def_with_body() {
            Some(def) => {
                def.diagnostics(db, &mut acc, style_lints);
            }
            None => {
                for diag in hir_ty::diagnostics::incorrect_case(db, id) {
                    acc.push(diag.into())
                }
            }
        }

        if let Some(def) = self.as_self_generic_def() {
            def.diagnostics(db, &mut acc);
        }

        acc
    }

    pub fn as_def_with_body(self) -> Option<DefWithBody> {
        match self {
            ModuleDef::Function(it) => Some(it.into()),
            ModuleDef::Const(it) => Some(it.into()),
            ModuleDef::Static(it) => Some(it.into()),
            ModuleDef::Variant(it) => Some(it.into()),

            ModuleDef::Module(_)
            | ModuleDef::Adt(_)
            | ModuleDef::Trait(_)
            | ModuleDef::TraitAlias(_)
            | ModuleDef::TypeAlias(_)
            | ModuleDef::Macro(_)
            | ModuleDef::BuiltinType(_) => None,
        }
    }

    /// Returns only defs that have generics from themselves, not their parent.
    pub fn as_self_generic_def(self) -> Option<GenericDef> {
        match self {
            ModuleDef::Function(it) => Some(it.into()),
            ModuleDef::Adt(it) => Some(it.into()),
            ModuleDef::Trait(it) => Some(it.into()),
            ModuleDef::TraitAlias(it) => Some(it.into()),
            ModuleDef::TypeAlias(it) => Some(it.into()),
            ModuleDef::Module(_)
            | ModuleDef::Variant(_)
            | ModuleDef::Static(_)
            | ModuleDef::Const(_)
            | ModuleDef::BuiltinType(_)
            | ModuleDef::Macro(_) => None,
        }
    }

    pub fn attrs(&self, db: &dyn HirDatabase) -> Option<AttrsWithOwner> {
        Some(match self {
            ModuleDef::Module(it) => it.attrs(db),
            ModuleDef::Function(it) => it.attrs(db),
            ModuleDef::Adt(it) => it.attrs(db),
            ModuleDef::Variant(it) => it.attrs(db),
            ModuleDef::Const(it) => it.attrs(db),
            ModuleDef::Static(it) => it.attrs(db),
            ModuleDef::Trait(it) => it.attrs(db),
            ModuleDef::TraitAlias(it) => it.attrs(db),
            ModuleDef::TypeAlias(it) => it.attrs(db),
            ModuleDef::Macro(it) => it.attrs(db),
            ModuleDef::BuiltinType(_) => return None,
        })
    }
}

impl HasCrate for ModuleDef {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        match self.module(db) {
            Some(module) => module.krate(),
            None => Crate::core(db).unwrap_or_else(|| db.all_crates()[0].into()),
        }
    }
}

impl HasVisibility for ModuleDef {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        match *self {
            ModuleDef::Module(it) => it.visibility(db),
            ModuleDef::Function(it) => it.visibility(db),
            ModuleDef::Adt(it) => it.visibility(db),
            ModuleDef::Const(it) => it.visibility(db),
            ModuleDef::Static(it) => it.visibility(db),
            ModuleDef::Trait(it) => it.visibility(db),
            ModuleDef::TraitAlias(it) => it.visibility(db),
            ModuleDef::TypeAlias(it) => it.visibility(db),
            ModuleDef::Variant(it) => it.visibility(db),
            ModuleDef::Macro(it) => it.visibility(db),
            ModuleDef::BuiltinType(_) => Visibility::Public,
        }
    }
}

impl Module {
    /// Name of this module.
    pub fn name(self, db: &dyn HirDatabase) -> Option<Name> {
        self.id.name(db)
    }

    /// Returns the crate this module is part of.
    pub fn krate(self) -> Crate {
        Crate { id: self.id.krate() }
    }

    /// Topmost parent of this module. Every module has a `crate_root`, but some
    /// might be missing `krate`. This can happen if a module's file is not included
    /// in the module tree of any target in `Cargo.toml`.
    pub fn crate_root(self, db: &dyn HirDatabase) -> Module {
        let def_map = crate_def_map(db, self.id.krate());
        Module { id: def_map.crate_root().into() }
    }

    pub fn is_crate_root(self) -> bool {
        DefMap::ROOT == self.id.local_id
    }

    /// Iterates over all child modules.
    pub fn children(self, db: &dyn HirDatabase) -> impl Iterator<Item = Module> {
        let def_map = self.id.def_map(db);
        let children = def_map[self.id.local_id]
            .children
            .values()
            .map(|module_id| Module { id: def_map.module_id(*module_id) })
            .collect::<Vec<_>>();
        children.into_iter()
    }

    /// Finds a parent module.
    pub fn parent(self, db: &dyn HirDatabase) -> Option<Module> {
        let def_map = self.id.def_map(db);
        let parent_id = def_map.containing_module(self.id.local_id)?;
        Some(Module { id: parent_id })
    }

    /// Finds nearest non-block ancestor `Module` (`self` included).
    pub fn nearest_non_block_module(self, db: &dyn HirDatabase) -> Module {
        let mut id = self.id;
        while id.is_block_module() {
            id = id.containing_module(db).expect("block without parent module");
        }
        Module { id }
    }

    pub fn path_to_root(self, db: &dyn HirDatabase) -> Vec<Module> {
        let mut res = vec![self];
        let mut curr = self;
        while let Some(next) = curr.parent(db) {
            res.push(next);
            curr = next
        }
        res
    }

    /// Returns a `ModuleScope`: a set of items, visible in this module.
    pub fn scope(
        self,
        db: &dyn HirDatabase,
        visible_from: Option<Module>,
    ) -> Vec<(Name, ScopeDef)> {
        self.id.def_map(db)[self.id.local_id]
            .scope
            .entries()
            .filter_map(|(name, def)| {
                if let Some(m) = visible_from {
                    let filtered = def.filter_visibility(|vis| vis.is_visible_from(db, m.id));
                    if filtered.is_none() && !def.is_none() { None } else { Some((name, filtered)) }
                } else {
                    Some((name, def))
                }
            })
            .flat_map(|(name, def)| {
                ScopeDef::all_items(def).into_iter().map(move |item| (name.clone(), item))
            })
            .collect()
    }

    pub fn resolve_mod_path(
        &self,
        db: &dyn HirDatabase,
        segments: impl IntoIterator<Item = Name>,
    ) -> Option<impl Iterator<Item = ItemInNs>> {
        let items = self
            .id
            .resolver(db)
            .resolve_module_path_in_items(db, &ModPath::from_segments(PathKind::Plain, segments));
        Some(items.iter_items().map(|(item, _)| item.into()))
    }

    /// Fills `acc` with the module's diagnostics.
    pub fn diagnostics<'db>(
        self,
        db: &'db dyn HirDatabase,
        acc: &mut Vec<AnyDiagnostic<'db>>,
        style_lints: bool,
    ) {
        let _p = tracing::info_span!("diagnostics", name = ?self.name(db)).entered();
        let edition = self.id.krate().data(db).edition;
        let def_map = self.id.def_map(db);
        for diag in def_map.diagnostics() {
            if diag.in_module != self.id.local_id {
                // FIXME: This is accidentally quadratic.
                continue;
            }
            emit_def_diagnostic(db, acc, diag, edition);
        }

        if !self.id.is_block_module() {
            // These are reported by the body of block modules
            let scope = &def_map[self.id.local_id].scope;
            scope.all_macro_calls().for_each(|it| macro_call_diagnostics(db, it, acc));
        }

        for def in self.declarations(db) {
            match def {
                ModuleDef::Module(m) => {
                    // Only add diagnostics from inline modules
                    if def_map[m.id.local_id].origin.is_inline() {
                        m.diagnostics(db, acc, style_lints)
                    }
                    acc.extend(def.diagnostics(db, style_lints))
                }
                ModuleDef::Trait(t) => {
                    for diag in TraitItems::query_with_diagnostics(db, t.id).1.iter() {
                        emit_def_diagnostic(db, acc, diag, edition);
                    }

                    for item in t.items(db) {
                        item.diagnostics(db, acc, style_lints);
                    }

                    t.all_macro_calls(db)
                        .iter()
                        .for_each(|&(_ast, call_id)| macro_call_diagnostics(db, call_id, acc));

                    acc.extend(def.diagnostics(db, style_lints))
                }
                ModuleDef::Adt(adt) => {
                    match adt {
                        Adt::Struct(s) => {
                            let source_map = db.struct_signature_with_source_map(s.id).1;
                            expr_store_diagnostics(db, acc, &source_map);
                            let source_map = &s.id.fields_with_source_map(db).1;
                            expr_store_diagnostics(db, acc, source_map);
                            push_ty_diagnostics(
                                db,
                                acc,
                                db.field_types_with_diagnostics(s.id.into()).1,
                                source_map,
                            );
                        }
                        Adt::Union(u) => {
                            let source_map = db.union_signature_with_source_map(u.id).1;
                            expr_store_diagnostics(db, acc, &source_map);
                            let source_map = &u.id.fields_with_source_map(db).1;
                            expr_store_diagnostics(db, acc, source_map);
                            push_ty_diagnostics(
                                db,
                                acc,
                                db.field_types_with_diagnostics(u.id.into()).1,
                                source_map,
                            );
                        }
                        Adt::Enum(e) => {
                            let source_map = db.enum_signature_with_source_map(e.id).1;
                            expr_store_diagnostics(db, acc, &source_map);
                            let (variants, diagnostics) = e.id.enum_variants_with_diagnostics(db);
                            let file = e.id.lookup(db).id.file_id;
                            let ast_id_map = db.ast_id_map(file);
                            if let Some(diagnostics) = &diagnostics {
                                for diag in diagnostics.iter() {
                                    acc.push(
                                        InactiveCode {
                                            node: InFile::new(
                                                file,
                                                ast_id_map.get(diag.ast_id).syntax_node_ptr(),
                                            ),
                                            cfg: diag.cfg.clone(),
                                            opts: diag.opts.clone(),
                                        }
                                        .into(),
                                    );
                                }
                            }
                            for &(v, _, _) in &variants.variants {
                                let source_map = &v.fields_with_source_map(db).1;
                                push_ty_diagnostics(
                                    db,
                                    acc,
                                    db.field_types_with_diagnostics(v.into()).1,
                                    source_map,
                                );
                                expr_store_diagnostics(db, acc, source_map);
                            }
                        }
                    }
                    acc.extend(def.diagnostics(db, style_lints))
                }
                ModuleDef::Macro(m) => emit_macro_def_diagnostics(db, acc, m),
                ModuleDef::TypeAlias(type_alias) => {
                    let source_map = db.type_alias_signature_with_source_map(type_alias.id).1;
                    expr_store_diagnostics(db, acc, &source_map);
                    push_ty_diagnostics(
                        db,
                        acc,
                        db.type_for_type_alias_with_diagnostics(type_alias.id).1,
                        &source_map,
                    );
                    acc.extend(def.diagnostics(db, style_lints));
                }
                _ => acc.extend(def.diagnostics(db, style_lints)),
            }
        }
        self.legacy_macros(db).into_iter().for_each(|m| emit_macro_def_diagnostics(db, acc, m));

        let inherent_impls = db.inherent_impls_in_crate(self.id.krate());

        let mut impl_assoc_items_scratch = vec![];
        for impl_def in self.impl_defs(db) {
            GenericDef::Impl(impl_def).diagnostics(db, acc);

            let loc = impl_def.id.lookup(db);
            let source_map = db.impl_signature_with_source_map(impl_def.id).1;
            expr_store_diagnostics(db, acc, &source_map);

            let file_id = loc.id.file_id;
            if file_id.macro_file().is_some_and(|it| it.kind(db) == MacroKind::DeriveBuiltIn) {
                // these expansion come from us, diagnosing them is a waste of resources
                // FIXME: Once we diagnose the inputs to builtin derives, we should at least extract those diagnostics somehow
                continue;
            }
            impl_def
                .all_macro_calls(db)
                .iter()
                .for_each(|&(_ast, call_id)| macro_call_diagnostics(db, call_id, acc));

            let ast_id_map = db.ast_id_map(file_id);

            for diag in impl_def.id.impl_items_with_diagnostics(db).1.iter() {
                emit_def_diagnostic(db, acc, diag, edition);
            }

            if inherent_impls.invalid_impls().contains(&impl_def.id) {
                acc.push(IncoherentImpl { impl_: ast_id_map.get(loc.id.value), file_id }.into())
            }

            if !impl_def.check_orphan_rules(db) {
                acc.push(TraitImplOrphan { impl_: ast_id_map.get(loc.id.value), file_id }.into())
            }

            let trait_ = impl_def.trait_(db);
            let trait_is_unsafe = trait_.is_some_and(|t| t.is_unsafe(db));
            let impl_is_negative = impl_def.is_negative(db);
            let impl_is_unsafe = impl_def.is_unsafe(db);

            let drop_maybe_dangle = (|| {
                // FIXME: This can be simplified a lot by exposing hir-ty's utils.rs::Generics helper
                let trait_ = trait_?;
                let drop_trait = LangItem::Drop.resolve_trait(db, self.krate().into())?;
                if drop_trait != trait_.into() {
                    return None;
                }
                let parent = impl_def.id.into();
                let generic_params = db.generic_params(parent);
                let lifetime_params = generic_params.iter_lt().map(|(local_id, _)| {
                    GenericParamId::LifetimeParamId(LifetimeParamId { parent, local_id })
                });
                let type_params = generic_params
                    .iter_type_or_consts()
                    .filter(|(_, it)| it.type_param().is_some())
                    .map(|(local_id, _)| {
                        GenericParamId::TypeParamId(TypeParamId::from_unchecked(
                            TypeOrConstParamId { parent, local_id },
                        ))
                    });
                let res = type_params.chain(lifetime_params).any(|p| {
                    db.attrs(AttrDefId::GenericParamId(p)).by_key(sym::may_dangle).exists()
                });
                Some(res)
            })()
            .unwrap_or(false);

            match (impl_is_unsafe, trait_is_unsafe, impl_is_negative, drop_maybe_dangle) {
                // unsafe negative impl
                (true, _, true, _) |
                // unsafe impl for safe trait
                (true, false, _, false) => acc.push(TraitImplIncorrectSafety { impl_: ast_id_map.get(loc.id.value), file_id, should_be_safe: true }.into()),
                // safe impl for unsafe trait
                (false, true, false, _) |
                // safe impl of dangling drop
                (false, false, _, true) => acc.push(TraitImplIncorrectSafety { impl_: ast_id_map.get(loc.id.value), file_id, should_be_safe: false }.into()),
                _ => (),
            };

            // Negative impls can't have items, don't emit missing items diagnostic for them
            if let (false, Some(trait_)) = (impl_is_negative, trait_) {
                let items = &trait_.id.trait_items(db).items;
                let required_items = items.iter().filter(|&(_, assoc)| match *assoc {
                    AssocItemId::FunctionId(it) => !db.function_signature(it).has_body(),
                    AssocItemId::ConstId(id) => !db.const_signature(id).has_body(),
                    AssocItemId::TypeAliasId(it) => db.type_alias_signature(it).ty.is_none(),
                });
                impl_assoc_items_scratch.extend(impl_def.id.impl_items(db).items.iter().cloned());

                let redundant = impl_assoc_items_scratch
                    .iter()
                    .filter(|(name, id)| {
                        !items.iter().any(|(impl_name, impl_item)| {
                            discriminant(impl_item) == discriminant(id) && impl_name == name
                        })
                    })
                    .map(|(name, item)| (name.clone(), AssocItem::from(*item)));
                for (name, assoc_item) in redundant {
                    acc.push(
                        TraitImplRedundantAssocItems {
                            trait_,
                            file_id,
                            impl_: ast_id_map.get(loc.id.value),
                            assoc_item: (name, assoc_item),
                        }
                        .into(),
                    )
                }

                let mut missing: Vec<_> = required_items
                    .filter(|(name, id)| {
                        !impl_assoc_items_scratch.iter().any(|(impl_name, impl_item)| {
                            discriminant(impl_item) == discriminant(id) && impl_name == name
                        })
                    })
                    .map(|(name, item)| (name.clone(), AssocItem::from(*item)))
                    .collect();

                if !missing.is_empty() {
                    let self_ty = db.impl_self_ty(impl_def.id).substitute(
                        Interner,
                        &hir_ty::generics::generics(db, impl_def.id.into()).placeholder_subst(db),
                    );
                    let self_ty = if let TyKind::Alias(AliasTy::Projection(projection)) =
                        self_ty.kind(Interner)
                    {
                        db.normalize_projection(
                            projection.clone(),
                            db.trait_environment(impl_def.id.into()),
                        )
                    } else {
                        self_ty
                    };
                    let self_ty_is_guaranteed_unsized = matches!(
                        self_ty.kind(Interner),
                        TyKind::Dyn(..) | TyKind::Slice(..) | TyKind::Str
                    );
                    if self_ty_is_guaranteed_unsized {
                        missing.retain(|(_, assoc_item)| {
                            let assoc_item = match *assoc_item {
                                AssocItem::Function(it) => it.id.into(),
                                AssocItem::Const(it) => it.id.into(),
                                AssocItem::TypeAlias(it) => it.id.into(),
                            };
                            !hir_ty::dyn_compatibility::generics_require_sized_self(db, assoc_item)
                        });
                    }
                }

                if !missing.is_empty() {
                    acc.push(
                        TraitImplMissingAssocItems {
                            impl_: ast_id_map.get(loc.id.value),
                            file_id,
                            missing,
                        }
                        .into(),
                    )
                }
                impl_assoc_items_scratch.clear();
            }

            push_ty_diagnostics(
                db,
                acc,
                db.impl_self_ty_with_diagnostics(impl_def.id).1,
                &source_map,
            );
            push_ty_diagnostics(
                db,
                acc,
                db.impl_trait_with_diagnostics(impl_def.id).and_then(|it| it.1),
                &source_map,
            );

            for &(_, item) in impl_def.id.impl_items(db).items.iter() {
                AssocItem::from(item).diagnostics(db, acc, style_lints);
            }
        }
    }

    pub fn declarations(self, db: &dyn HirDatabase) -> Vec<ModuleDef> {
        let def_map = self.id.def_map(db);
        let scope = &def_map[self.id.local_id].scope;
        scope
            .declarations()
            .map(ModuleDef::from)
            .chain(scope.unnamed_consts().map(|id| ModuleDef::Const(Const::from(id))))
            .collect()
    }

    pub fn legacy_macros(self, db: &dyn HirDatabase) -> Vec<Macro> {
        let def_map = self.id.def_map(db);
        let scope = &def_map[self.id.local_id].scope;
        scope.legacy_macros().flat_map(|(_, it)| it).map(|&it| it.into()).collect()
    }

    pub fn impl_defs(self, db: &dyn HirDatabase) -> Vec<Impl> {
        let def_map = self.id.def_map(db);
        def_map[self.id.local_id].scope.impls().map(Impl::from).collect()
    }

    /// Finds a path that can be used to refer to the given item from within
    /// this module, if possible.
    pub fn find_path(
        self,
        db: &dyn DefDatabase,
        item: impl Into<ItemInNs>,
        cfg: ImportPathConfig,
    ) -> Option<ModPath> {
        hir_def::find_path::find_path(
            db,
            item.into().into(),
            self.into(),
            PrefixKind::Plain,
            false,
            cfg,
        )
    }

    /// Finds a path that can be used to refer to the given item from within
    /// this module, if possible. This is used for returning import paths for use-statements.
    pub fn find_use_path(
        self,
        db: &dyn DefDatabase,
        item: impl Into<ItemInNs>,
        prefix_kind: PrefixKind,
        cfg: ImportPathConfig,
    ) -> Option<ModPath> {
        hir_def::find_path::find_path(db, item.into().into(), self.into(), prefix_kind, true, cfg)
    }
}

fn macro_call_diagnostics<'db>(
    db: &'db dyn HirDatabase,
    macro_call_id: MacroCallId,
    acc: &mut Vec<AnyDiagnostic<'db>>,
) {
    let Some(e) = db.parse_macro_expansion_error(macro_call_id) else {
        return;
    };
    let ValueResult { value: parse_errors, err } = &*e;
    if let Some(err) = err {
        let loc = db.lookup_intern_macro_call(macro_call_id);
        let file_id = loc.kind.file_id();
        let node =
            InFile::new(file_id, db.ast_id_map(file_id).get_erased(loc.kind.erased_ast_id()));
        let RenderedExpandError { message, error, kind } = err.render_to_string(db);
        let editioned_file_id = EditionedFileId::from_span(db, err.span().anchor.file_id);
        let precise_location = if editioned_file_id == file_id {
            Some(
                err.span().range
                    + db.ast_id_map(editioned_file_id.into())
                        .get_erased(err.span().anchor.ast_id)
                        .text_range()
                        .start(),
            )
        } else {
            None
        };
        acc.push(MacroError { node, precise_location, message, error, kind }.into());
    }

    if !parse_errors.is_empty() {
        let loc = db.lookup_intern_macro_call(macro_call_id);
        let (node, precise_location) = precise_macro_call_location(&loc.kind, db);
        acc.push(
            MacroExpansionParseError { node, precise_location, errors: parse_errors.clone() }
                .into(),
        )
    }
}

fn emit_macro_def_diagnostics<'db>(
    db: &'db dyn HirDatabase,
    acc: &mut Vec<AnyDiagnostic<'db>>,
    m: Macro,
) {
    let id = db.macro_def(m.id);
    if let hir_expand::db::TokenExpander::DeclarativeMacro(expander) = db.macro_expander(id) {
        if let Some(e) = expander.mac.err() {
            let Some(ast) = id.ast_id().left() else {
                never!("declarative expander for non decl-macro: {:?}", e);
                return;
            };
            let krate = HasModule::krate(&m.id, db);
            let edition = krate.data(db).edition;
            emit_def_diagnostic_(
                db,
                acc,
                &DefDiagnosticKind::MacroDefError { ast, message: e.to_string() },
                edition,
            );
        }
    }
}

fn emit_def_diagnostic<'db>(
    db: &'db dyn HirDatabase,
    acc: &mut Vec<AnyDiagnostic<'db>>,
    diag: &DefDiagnostic,
    edition: Edition,
) {
    emit_def_diagnostic_(db, acc, &diag.kind, edition)
}

fn emit_def_diagnostic_<'db>(
    db: &'db dyn HirDatabase,
    acc: &mut Vec<AnyDiagnostic<'db>>,
    diag: &DefDiagnosticKind,
    edition: Edition,
) {
    match diag {
        DefDiagnosticKind::UnresolvedModule { ast: declaration, candidates } => {
            let decl = declaration.to_ptr(db);
            acc.push(
                UnresolvedModule {
                    decl: InFile::new(declaration.file_id, decl),
                    candidates: candidates.clone(),
                }
                .into(),
            )
        }
        DefDiagnosticKind::UnresolvedExternCrate { ast } => {
            let item = ast.to_ptr(db);
            acc.push(UnresolvedExternCrate { decl: InFile::new(ast.file_id, item) }.into());
        }

        DefDiagnosticKind::MacroError { ast, path, err } => {
            let item = ast.to_ptr(db);
            let RenderedExpandError { message, error, kind } = err.render_to_string(db);
            acc.push(
                MacroError {
                    node: InFile::new(ast.file_id, item.syntax_node_ptr()),
                    precise_location: None,
                    message: format!("{}: {message}", path.display(db, edition)),
                    error,
                    kind,
                }
                .into(),
            )
        }
        DefDiagnosticKind::UnresolvedImport { id, index } => {
            let file_id = id.file_id;

            let use_tree = hir_def::src::use_tree_to_ast(db, *id, *index);
            acc.push(
                UnresolvedImport { decl: InFile::new(file_id, AstPtr::new(&use_tree)) }.into(),
            );
        }

        DefDiagnosticKind::UnconfiguredCode { ast_id, cfg, opts } => {
            let ast_id_map = db.ast_id_map(ast_id.file_id);
            let ptr = ast_id_map.get_erased(ast_id.value);
            acc.push(
                InactiveCode {
                    node: InFile::new(ast_id.file_id, ptr),
                    cfg: cfg.clone(),
                    opts: opts.clone(),
                }
                .into(),
            );
        }
        DefDiagnosticKind::UnresolvedMacroCall { ast, path } => {
            let (node, precise_location) = precise_macro_call_location(ast, db);
            acc.push(
                UnresolvedMacroCall {
                    macro_call: node,
                    precise_location,
                    path: path.clone(),
                    is_bang: matches!(ast, MacroCallKind::FnLike { .. }),
                }
                .into(),
            );
        }
        DefDiagnosticKind::UnimplementedBuiltinMacro { ast } => {
            let node = ast.to_node(db);
            // Must have a name, otherwise we wouldn't emit it.
            let name = node.name().expect("unimplemented builtin macro with no name");
            acc.push(
                UnimplementedBuiltinMacro {
                    node: ast.with_value(SyntaxNodePtr::from(AstPtr::new(&name))),
                }
                .into(),
            );
        }
        DefDiagnosticKind::InvalidDeriveTarget { ast, id } => {
            let node = ast.to_node(db);
            let derive = node.attrs().nth(*id);
            match derive {
                Some(derive) => {
                    acc.push(
                        InvalidDeriveTarget {
                            node: ast.with_value(SyntaxNodePtr::from(AstPtr::new(&derive))),
                        }
                        .into(),
                    );
                }
                None => stdx::never!("derive diagnostic on item without derive attribute"),
            }
        }
        DefDiagnosticKind::MalformedDerive { ast, id } => {
            let node = ast.to_node(db);
            let derive = node.attrs().nth(*id);
            match derive {
                Some(derive) => {
                    acc.push(
                        MalformedDerive {
                            node: ast.with_value(SyntaxNodePtr::from(AstPtr::new(&derive))),
                        }
                        .into(),
                    );
                }
                None => stdx::never!("derive diagnostic on item without derive attribute"),
            }
        }
        DefDiagnosticKind::MacroDefError { ast, message } => {
            let node = ast.to_node(db);
            acc.push(
                MacroDefError {
                    node: InFile::new(ast.file_id, AstPtr::new(&node)),
                    name: node.name().map(|it| it.syntax().text_range()),
                    message: message.clone(),
                }
                .into(),
            );
        }
    }
}

fn precise_macro_call_location(
    ast: &MacroCallKind,
    db: &dyn HirDatabase,
) -> (InFile<SyntaxNodePtr>, Option<TextRange>) {
    // FIXME: maybe we actually want slightly different ranges for the different macro diagnostics
    // - e.g. the full attribute for macro errors, but only the name for name resolution
    match ast {
        MacroCallKind::FnLike { ast_id, .. } => {
            let node = ast_id.to_node(db);
            (
                ast_id.with_value(SyntaxNodePtr::from(AstPtr::new(&node))),
                node.path()
                    .and_then(|it| it.segment())
                    .and_then(|it| it.name_ref())
                    .map(|it| it.syntax().text_range()),
            )
        }
        MacroCallKind::Derive { ast_id, derive_attr_index, derive_index, .. } => {
            let node = ast_id.to_node(db);
            // Compute the precise location of the macro name's token in the derive
            // list.
            let token = (|| {
                let derive_attr = collect_attrs(&node)
                    .nth(derive_attr_index.ast_index())
                    .and_then(|x| Either::left(x.1))?;
                let token_tree = derive_attr.meta()?.token_tree()?;
                let chunk_by = token_tree
                    .syntax()
                    .children_with_tokens()
                    .filter_map(|elem| match elem {
                        syntax::NodeOrToken::Token(tok) => Some(tok),
                        _ => None,
                    })
                    .chunk_by(|t| t.kind() == T![,]);
                let (_, mut group) = chunk_by
                    .into_iter()
                    .filter(|&(comma, _)| !comma)
                    .nth(*derive_index as usize)?;
                group.find(|t| t.kind() == T![ident])
            })();
            (
                ast_id.with_value(SyntaxNodePtr::from(AstPtr::new(&node))),
                token.as_ref().map(|tok| tok.text_range()),
            )
        }
        MacroCallKind::Attr { ast_id, invoc_attr_index, .. } => {
            let node = ast_id.to_node(db);
            let attr = collect_attrs(&node)
                .nth(invoc_attr_index.ast_index())
                .and_then(|x| Either::left(x.1))
                .unwrap_or_else(|| {
                    panic!("cannot find attribute #{}", invoc_attr_index.ast_index())
                });

            (
                ast_id.with_value(SyntaxNodePtr::from(AstPtr::new(&attr))),
                Some(attr.syntax().text_range()),
            )
        }
    }
}

impl HasVisibility for Module {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        let def_map = self.id.def_map(db);
        let module_data = &def_map[self.id.local_id];
        module_data.visibility
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Field {
    pub(crate) parent: VariantDef,
    pub(crate) id: LocalFieldId,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub struct TupleField {
    pub owner: DefWithBodyId,
    pub tuple: TupleId,
    pub index: u32,
}

impl TupleField {
    pub fn name(&self) -> Name {
        Name::new_tuple_field(self.index as usize)
    }

    pub fn ty<'db>(&self, db: &'db dyn HirDatabase) -> Type<'db> {
        let ty = db.infer(self.owner).tuple_field_access_types[&self.tuple]
            .as_slice(Interner)
            .get(self.index as usize)
            .and_then(|arg| arg.ty(Interner))
            .cloned()
            .unwrap_or_else(|| TyKind::Error.intern(Interner));
        Type {
            env: db.trait_environment_for_body(self.owner),
            ty,
            _pd: PhantomCovariantLifetime::new(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum FieldSource {
    Named(ast::RecordField),
    Pos(ast::TupleField),
}

impl AstNode for FieldSource {
    fn can_cast(kind: syntax::SyntaxKind) -> bool
    where
        Self: Sized,
    {
        ast::RecordField::can_cast(kind) || ast::TupleField::can_cast(kind)
    }

    fn cast(syntax: SyntaxNode) -> Option<Self>
    where
        Self: Sized,
    {
        if ast::RecordField::can_cast(syntax.kind()) {
            <ast::RecordField as AstNode>::cast(syntax).map(FieldSource::Named)
        } else if ast::TupleField::can_cast(syntax.kind()) {
            <ast::TupleField as AstNode>::cast(syntax).map(FieldSource::Pos)
        } else {
            None
        }
    }

    fn syntax(&self) -> &SyntaxNode {
        match self {
            FieldSource::Named(it) => it.syntax(),
            FieldSource::Pos(it) => it.syntax(),
        }
    }
}

impl Field {
    pub fn name(&self, db: &dyn HirDatabase) -> Name {
        VariantId::from(self.parent).fields(db).fields()[self.id].name.clone()
    }

    pub fn index(&self) -> usize {
        u32::from(self.id.into_raw()) as usize
    }

    /// Returns the type as in the signature of the struct (i.e., with
    /// placeholder types for type parameters). Only use this in the context of
    /// the field definition.
    pub fn ty<'db>(&self, db: &'db dyn HirDatabase) -> Type<'db> {
        let var_id = self.parent.into();
        let generic_def_id: GenericDefId = match self.parent {
            VariantDef::Struct(it) => it.id.into(),
            VariantDef::Union(it) => it.id.into(),
            VariantDef::Variant(it) => it.id.lookup(db).parent.into(),
        };
        let substs = TyBuilder::placeholder_subst(db, generic_def_id);
        let ty = db.field_types(var_id)[self.id].clone().substitute(Interner, &substs);
        Type::new(db, var_id, ty)
    }

    // FIXME: Find better API to also handle const generics
    pub fn ty_with_args<'db>(
        &self,
        db: &'db dyn HirDatabase,
        generics: impl Iterator<Item = Type<'db>>,
    ) -> Type<'db> {
        let var_id = self.parent.into();
        let def_id: AdtId = match self.parent {
            VariantDef::Struct(it) => it.id.into(),
            VariantDef::Union(it) => it.id.into(),
            VariantDef::Variant(it) => it.parent_enum(db).id.into(),
        };
        let mut generics = generics.map(|it| it.ty);
        let substs = TyBuilder::subst_for_def(db, def_id, None)
            .fill(|x| match x {
                ParamKind::Type => {
                    generics.next().unwrap_or_else(|| TyKind::Error.intern(Interner)).cast(Interner)
                }
                ParamKind::Const(ty) => unknown_const_as_generic(ty.clone()),
                ParamKind::Lifetime => error_lifetime().cast(Interner),
            })
            .build();
        let ty = db.field_types(var_id)[self.id].clone().substitute(Interner, &substs);
        Type::new(db, var_id, ty)
    }

    pub fn layout(&self, db: &dyn HirDatabase) -> Result<Layout, LayoutError> {
        db.layout_of_ty(
            self.ty(db).ty,
            db.trait_environment(match hir_def::VariantId::from(self.parent) {
                hir_def::VariantId::EnumVariantId(id) => {
                    GenericDefId::AdtId(id.lookup(db).parent.into())
                }
                hir_def::VariantId::StructId(id) => GenericDefId::AdtId(id.into()),
                hir_def::VariantId::UnionId(id) => GenericDefId::AdtId(id.into()),
            }),
        )
        .map(|layout| Layout(layout, db.target_data_layout(self.krate(db).into()).unwrap()))
    }

    pub fn parent_def(&self, _db: &dyn HirDatabase) -> VariantDef {
        self.parent
    }
}

impl HasVisibility for Field {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        let variant_data = VariantId::from(self.parent).fields(db);
        let visibility = &variant_data.fields()[self.id].visibility;
        let parent_id: hir_def::VariantId = self.parent.into();
        // FIXME: RawVisibility::Public doesn't need to construct a resolver
        Visibility::resolve(db, &parent_id.resolver(db), visibility)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Struct {
    pub(crate) id: StructId,
}

impl Struct {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        Module { id: self.id.lookup(db).container }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        db.struct_signature(self.id).name.clone()
    }

    pub fn fields(self, db: &dyn HirDatabase) -> Vec<Field> {
        self.id
            .fields(db)
            .fields()
            .iter()
            .map(|(id, _)| Field { parent: self.into(), id })
            .collect()
    }

    pub fn ty(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::from_def(db, self.id)
    }

    pub fn ty_placeholders(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::from_def_placeholders(db, self.id)
    }

    pub fn constructor_ty(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::from_value_def(db, self.id)
    }

    pub fn repr(self, db: &dyn HirDatabase) -> Option<ReprOptions> {
        db.struct_signature(self.id).repr
    }

    pub fn kind(self, db: &dyn HirDatabase) -> StructKind {
        match self.variant_fields(db).shape {
            hir_def::item_tree::FieldsShape::Record => StructKind::Record,
            hir_def::item_tree::FieldsShape::Tuple => StructKind::Tuple,
            hir_def::item_tree::FieldsShape::Unit => StructKind::Unit,
        }
    }

    fn variant_fields(self, db: &dyn HirDatabase) -> &VariantFields {
        self.id.fields(db)
    }

    pub fn is_unstable(self, db: &dyn HirDatabase) -> bool {
        db.attrs(self.id.into()).is_unstable()
    }
}

impl HasVisibility for Struct {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        let loc = self.id.lookup(db);
        let source = loc.source(db);
        visibility_from_ast(db, self.id, source.map(|src| src.visibility()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Union {
    pub(crate) id: UnionId,
}

impl Union {
    pub fn name(self, db: &dyn HirDatabase) -> Name {
        db.union_signature(self.id).name.clone()
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        Module { id: self.id.lookup(db).container }
    }

    pub fn ty(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::from_def(db, self.id)
    }

    pub fn ty_placeholders(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::from_def_placeholders(db, self.id)
    }

    pub fn constructor_ty(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::from_value_def(db, self.id)
    }

    pub fn kind(self, db: &dyn HirDatabase) -> StructKind {
        match self.id.fields(db).shape {
            hir_def::item_tree::FieldsShape::Record => StructKind::Record,
            hir_def::item_tree::FieldsShape::Tuple => StructKind::Tuple,
            hir_def::item_tree::FieldsShape::Unit => StructKind::Unit,
        }
    }

    pub fn fields(self, db: &dyn HirDatabase) -> Vec<Field> {
        self.id
            .fields(db)
            .fields()
            .iter()
            .map(|(id, _)| Field { parent: self.into(), id })
            .collect()
    }
    pub fn is_unstable(self, db: &dyn HirDatabase) -> bool {
        db.attrs(self.id.into()).is_unstable()
    }
}

impl HasVisibility for Union {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        let loc = self.id.lookup(db);
        let source = loc.source(db);
        visibility_from_ast(db, self.id, source.map(|src| src.visibility()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Enum {
    pub(crate) id: EnumId,
}

impl Enum {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        Module { id: self.id.lookup(db).container }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        db.enum_signature(self.id).name.clone()
    }

    pub fn variants(self, db: &dyn HirDatabase) -> Vec<Variant> {
        self.id.enum_variants(db).variants.iter().map(|&(id, _, _)| Variant { id }).collect()
    }

    pub fn num_variants(self, db: &dyn HirDatabase) -> usize {
        self.id.enum_variants(db).variants.len()
    }

    pub fn repr(self, db: &dyn HirDatabase) -> Option<ReprOptions> {
        db.enum_signature(self.id).repr
    }

    pub fn ty<'db>(self, db: &'db dyn HirDatabase) -> Type<'db> {
        Type::from_def(db, self.id)
    }

    pub fn ty_placeholders<'db>(self, db: &'db dyn HirDatabase) -> Type<'db> {
        Type::from_def_placeholders(db, self.id)
    }

    /// The type of the enum variant bodies.
    pub fn variant_body_ty<'db>(self, db: &'db dyn HirDatabase) -> Type<'db> {
        Type::new_for_crate(
            self.id.lookup(db).container.krate(),
            TyBuilder::builtin(match db.enum_signature(self.id).variant_body_type() {
                layout::IntegerType::Pointer(sign) => match sign {
                    true => hir_def::builtin_type::BuiltinType::Int(
                        hir_def::builtin_type::BuiltinInt::Isize,
                    ),
                    false => hir_def::builtin_type::BuiltinType::Uint(
                        hir_def::builtin_type::BuiltinUint::Usize,
                    ),
                },
                layout::IntegerType::Fixed(i, sign) => match sign {
                    true => hir_def::builtin_type::BuiltinType::Int(match i {
                        layout::Integer::I8 => hir_def::builtin_type::BuiltinInt::I8,
                        layout::Integer::I16 => hir_def::builtin_type::BuiltinInt::I16,
                        layout::Integer::I32 => hir_def::builtin_type::BuiltinInt::I32,
                        layout::Integer::I64 => hir_def::builtin_type::BuiltinInt::I64,
                        layout::Integer::I128 => hir_def::builtin_type::BuiltinInt::I128,
                    }),
                    false => hir_def::builtin_type::BuiltinType::Uint(match i {
                        layout::Integer::I8 => hir_def::builtin_type::BuiltinUint::U8,
                        layout::Integer::I16 => hir_def::builtin_type::BuiltinUint::U16,
                        layout::Integer::I32 => hir_def::builtin_type::BuiltinUint::U32,
                        layout::Integer::I64 => hir_def::builtin_type::BuiltinUint::U64,
                        layout::Integer::I128 => hir_def::builtin_type::BuiltinUint::U128,
                    }),
                },
            }),
        )
    }

    /// Returns true if at least one variant of this enum is a non-unit variant.
    pub fn is_data_carrying(self, db: &dyn HirDatabase) -> bool {
        self.variants(db).iter().any(|v| !matches!(v.kind(db), StructKind::Unit))
    }

    pub fn layout(self, db: &dyn HirDatabase) -> Result<Layout, LayoutError> {
        Adt::from(self).layout(db)
    }

    pub fn is_unstable(self, db: &dyn HirDatabase) -> bool {
        db.attrs(self.id.into()).is_unstable()
    }
}

impl HasVisibility for Enum {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        let loc = self.id.lookup(db);
        let source = loc.source(db);
        visibility_from_ast(db, self.id, source.map(|src| src.visibility()))
    }
}

impl From<&Variant> for DefWithBodyId {
    fn from(&v: &Variant) -> Self {
        DefWithBodyId::VariantId(v.into())
    }
}

// FIXME: Rename to `EnumVariant`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variant {
    pub(crate) id: EnumVariantId,
}

impl Variant {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        Module { id: self.id.module(db) }
    }

    pub fn parent_enum(self, db: &dyn HirDatabase) -> Enum {
        self.id.lookup(db).parent.into()
    }

    pub fn constructor_ty(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::from_value_def(db, self.id)
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        let lookup = self.id.lookup(db);
        let enum_ = lookup.parent;
        enum_.enum_variants(db).variants[lookup.index as usize].1.clone()
    }

    pub fn fields(self, db: &dyn HirDatabase) -> Vec<Field> {
        self.id
            .fields(db)
            .fields()
            .iter()
            .map(|(id, _)| Field { parent: self.into(), id })
            .collect()
    }

    pub fn kind(self, db: &dyn HirDatabase) -> StructKind {
        match self.id.fields(db).shape {
            hir_def::item_tree::FieldsShape::Record => StructKind::Record,
            hir_def::item_tree::FieldsShape::Tuple => StructKind::Tuple,
            hir_def::item_tree::FieldsShape::Unit => StructKind::Unit,
        }
    }

    pub fn value(self, db: &dyn HirDatabase) -> Option<ast::Expr> {
        self.source(db)?.value.expr()
    }

    pub fn eval(self, db: &dyn HirDatabase) -> Result<i128, ConstEvalError> {
        db.const_eval_discriminant(self.into())
    }

    pub fn layout(&self, db: &dyn HirDatabase) -> Result<Layout, LayoutError> {
        let parent_enum = self.parent_enum(db);
        let parent_layout = parent_enum.layout(db)?;
        Ok(match &parent_layout.0.variants {
            layout::Variants::Multiple { variants, .. } => Layout(
                {
                    let lookup = self.id.lookup(db);
                    let rustc_enum_variant_idx = RustcEnumVariantIdx(lookup.index as usize);
                    Arc::new(variants[rustc_enum_variant_idx].clone())
                },
                db.target_data_layout(parent_enum.krate(db).into()).unwrap(),
            ),
            _ => parent_layout,
        })
    }

    pub fn is_unstable(self, db: &dyn HirDatabase) -> bool {
        db.attrs(self.id.into()).is_unstable()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructKind {
    Record,
    Tuple,
    Unit,
}

/// Variants inherit visibility from the parent enum.
impl HasVisibility for Variant {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        self.parent_enum(db).visibility(db)
    }
}

/// A Data Type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Adt {
    Struct(Struct),
    Union(Union),
    Enum(Enum),
}
impl_from!(Struct, Union, Enum for Adt);

impl Adt {
    pub fn has_non_default_type_params(self, db: &dyn HirDatabase) -> bool {
        let subst = db.generic_defaults(self.into());
        (subst.is_empty() && db.generic_params(self.into()).len_type_or_consts() != 0)
            || subst.iter().any(|ty| match ty.skip_binders().data(Interner) {
                GenericArgData::Ty(it) => it.is_unknown(),
                _ => false,
            })
    }

    pub fn layout(self, db: &dyn HirDatabase) -> Result<Layout, LayoutError> {
        db.layout_of_adt(
            self.into(),
            TyBuilder::adt(db, self.into())
                .fill_with_defaults(db, || TyKind::Error.intern(Interner))
                .build_into_subst(),
            db.trait_environment(self.into()),
        )
        .map(|layout| Layout(layout, db.target_data_layout(self.krate(db).id).unwrap()))
    }

    /// Turns this ADT into a type. Any type parameters of the ADT will be
    /// turned into unknown types, which is good for e.g. finding the most
    /// general set of completions, but will not look very nice when printed.
    pub fn ty(self, db: &dyn HirDatabase) -> Type<'_> {
        let id = AdtId::from(self);
        Type::from_def(db, id)
    }

    /// Turns this ADT into a type with the given type parameters. This isn't
    /// the greatest API, FIXME find a better one.
    pub fn ty_with_args<'db>(
        self,
        db: &'db dyn HirDatabase,
        args: impl IntoIterator<Item = Type<'db>>,
    ) -> Type<'db> {
        let id = AdtId::from(self);
        let mut it = args.into_iter().map(|t| t.ty);
        let ty = TyBuilder::def_ty(db, id.into(), None)
            .fill(|x| {
                let r = it.next().unwrap_or_else(|| TyKind::Error.intern(Interner));
                match x {
                    ParamKind::Type => r.cast(Interner),
                    ParamKind::Const(ty) => unknown_const_as_generic(ty.clone()),
                    ParamKind::Lifetime => error_lifetime().cast(Interner),
                }
            })
            .build();
        Type::new(db, id, ty)
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        match self {
            Adt::Struct(s) => s.module(db),
            Adt::Union(s) => s.module(db),
            Adt::Enum(e) => e.module(db),
        }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        match self {
            Adt::Struct(s) => s.name(db),
            Adt::Union(u) => u.name(db),
            Adt::Enum(e) => e.name(db),
        }
    }

    /// Returns the lifetime of the DataType
    pub fn lifetime(&self, db: &dyn HirDatabase) -> Option<LifetimeParamData> {
        let resolver = match self {
            Adt::Struct(s) => s.id.resolver(db),
            Adt::Union(u) => u.id.resolver(db),
            Adt::Enum(e) => e.id.resolver(db),
        };
        resolver
            .generic_params()
            .and_then(|gp| {
                gp.iter_lt()
                    // there should only be a single lifetime
                    // but `Arena` requires to use an iterator
                    .nth(0)
            })
            .map(|arena| arena.1.clone())
    }

    pub fn as_struct(&self) -> Option<Struct> {
        if let Self::Struct(v) = self { Some(*v) } else { None }
    }

    pub fn as_enum(&self) -> Option<Enum> {
        if let Self::Enum(v) = self { Some(*v) } else { None }
    }
}

impl HasVisibility for Adt {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        match self {
            Adt::Struct(it) => it.visibility(db),
            Adt::Union(it) => it.visibility(db),
            Adt::Enum(it) => it.visibility(db),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VariantDef {
    Struct(Struct),
    Union(Union),
    Variant(Variant),
}
impl_from!(Struct, Union, Variant for VariantDef);

impl VariantDef {
    pub fn fields(self, db: &dyn HirDatabase) -> Vec<Field> {
        match self {
            VariantDef::Struct(it) => it.fields(db),
            VariantDef::Union(it) => it.fields(db),
            VariantDef::Variant(it) => it.fields(db),
        }
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        match self {
            VariantDef::Struct(it) => it.module(db),
            VariantDef::Union(it) => it.module(db),
            VariantDef::Variant(it) => it.module(db),
        }
    }

    pub fn name(&self, db: &dyn HirDatabase) -> Name {
        match self {
            VariantDef::Struct(s) => s.name(db),
            VariantDef::Union(u) => u.name(db),
            VariantDef::Variant(e) => e.name(db),
        }
    }
}

/// The defs which have a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefWithBody {
    Function(Function),
    Static(Static),
    Const(Const),
    Variant(Variant),
}
impl_from!(Function, Const, Static, Variant for DefWithBody);

impl DefWithBody {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        match self {
            DefWithBody::Const(c) => c.module(db),
            DefWithBody::Function(f) => f.module(db),
            DefWithBody::Static(s) => s.module(db),
            DefWithBody::Variant(v) => v.module(db),
        }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Option<Name> {
        match self {
            DefWithBody::Function(f) => Some(f.name(db)),
            DefWithBody::Static(s) => Some(s.name(db)),
            DefWithBody::Const(c) => c.name(db),
            DefWithBody::Variant(v) => Some(v.name(db)),
        }
    }

    /// Returns the type this def's body has to evaluate to.
    pub fn body_type(self, db: &dyn HirDatabase) -> Type<'_> {
        match self {
            DefWithBody::Function(it) => it.ret_type(db),
            DefWithBody::Static(it) => it.ty(db),
            DefWithBody::Const(it) => it.ty(db),
            DefWithBody::Variant(it) => it.parent_enum(db).variant_body_ty(db),
        }
    }

    fn id(&self) -> DefWithBodyId {
        match self {
            DefWithBody::Function(it) => it.id.into(),
            DefWithBody::Static(it) => it.id.into(),
            DefWithBody::Const(it) => it.id.into(),
            DefWithBody::Variant(it) => it.into(),
        }
    }

    /// A textual representation of the HIR of this def's body for debugging purposes.
    pub fn debug_hir(self, db: &dyn HirDatabase) -> String {
        let body = db.body(self.id());
        body.pretty_print(db, self.id(), Edition::CURRENT)
    }

    /// A textual representation of the MIR of this def's body for debugging purposes.
    pub fn debug_mir(self, db: &dyn HirDatabase) -> String {
        let body = db.mir_body(self.id());
        match body {
            Ok(body) => body.pretty_print(db, self.module(db).krate().to_display_target(db)),
            Err(e) => format!("error:\n{e:?}"),
        }
    }

    pub fn diagnostics<'db>(
        self,
        db: &'db dyn HirDatabase,
        acc: &mut Vec<AnyDiagnostic<'db>>,
        style_lints: bool,
    ) {
        let krate = self.module(db).id.krate();

        let (body, source_map) = db.body_with_source_map(self.into());
        let sig_source_map = match self {
            DefWithBody::Function(id) => db.function_signature_with_source_map(id.into()).1,
            DefWithBody::Static(id) => db.static_signature_with_source_map(id.into()).1,
            DefWithBody::Const(id) => db.const_signature_with_source_map(id.into()).1,
            DefWithBody::Variant(variant) => {
                let enum_id = variant.parent_enum(db).id;
                db.enum_signature_with_source_map(enum_id).1
            }
        };

        for (_, def_map) in body.blocks(db) {
            Module { id: def_map.module_id(DefMap::ROOT) }.diagnostics(db, acc, style_lints);
        }

        source_map
            .macro_calls()
            .for_each(|(_ast_id, call_id)| macro_call_diagnostics(db, call_id, acc));

        expr_store_diagnostics(db, acc, &source_map);

        let infer = db.infer(self.into());
        for d in &infer.diagnostics {
            acc.extend(AnyDiagnostic::inference_diagnostic(
                db,
                self.into(),
                d,
                &source_map,
                &sig_source_map,
            ));
        }

        for (pat_or_expr, mismatch) in infer.type_mismatches() {
            let expr_or_pat = match pat_or_expr {
                ExprOrPatId::ExprId(expr) => source_map.expr_syntax(expr).map(Either::Left),
                ExprOrPatId::PatId(pat) => source_map.pat_syntax(pat).map(Either::Right),
            };
            let expr_or_pat = match expr_or_pat {
                Ok(Either::Left(expr)) => expr,
                Ok(Either::Right(InFile { file_id, value: pat })) => {
                    // cast from Either<Pat, SelfParam> -> Either<_, Pat>
                    let Some(ptr) = AstPtr::try_from_raw(pat.syntax_node_ptr()) else {
                        continue;
                    };
                    InFile { file_id, value: ptr }
                }
                Err(SyntheticSyntax) => continue,
            };

            acc.push(
                TypeMismatch {
                    expr_or_pat,
                    expected: Type::new(db, DefWithBodyId::from(self), mismatch.expected.clone()),
                    actual: Type::new(db, DefWithBodyId::from(self), mismatch.actual.clone()),
                }
                .into(),
            );
        }

        let missing_unsafe = hir_ty::diagnostics::missing_unsafe(db, self.into());
        for (node, reason) in missing_unsafe.unsafe_exprs {
            match source_map.expr_or_pat_syntax(node) {
                Ok(node) => acc.push(
                    MissingUnsafe {
                        node,
                        lint: if missing_unsafe.fn_is_unsafe {
                            UnsafeLint::UnsafeOpInUnsafeFn
                        } else {
                            UnsafeLint::HardError
                        },
                        reason,
                    }
                    .into(),
                ),
                Err(SyntheticSyntax) => {
                    // FIXME: Here and elsewhere in this file, the `expr` was
                    // desugared, report or assert that this doesn't happen.
                }
            }
        }
        for node in missing_unsafe.deprecated_safe_calls {
            match source_map.expr_syntax(node) {
                Ok(node) => acc.push(
                    MissingUnsafe {
                        node,
                        lint: UnsafeLint::DeprecatedSafe2024,
                        reason: UnsafetyReason::UnsafeFnCall,
                    }
                    .into(),
                ),
                Err(SyntheticSyntax) => never!("synthetic DeprecatedSafe2024"),
            }
        }

        if let Ok(borrowck_results) = db.borrowck(self.into()) {
            for borrowck_result in borrowck_results.iter() {
                let mir_body = &borrowck_result.mir_body;
                for moof in &borrowck_result.moved_out_of_ref {
                    let span: InFile<SyntaxNodePtr> = match moof.span {
                        mir::MirSpan::ExprId(e) => match source_map.expr_syntax(e) {
                            Ok(s) => s.map(|it| it.into()),
                            Err(_) => continue,
                        },
                        mir::MirSpan::PatId(p) => match source_map.pat_syntax(p) {
                            Ok(s) => s.map(|it| it.into()),
                            Err(_) => continue,
                        },
                        mir::MirSpan::SelfParam => match source_map.self_param_syntax() {
                            Some(s) => s.map(|it| it.into()),
                            None => continue,
                        },
                        mir::MirSpan::BindingId(b) => {
                            match source_map
                                .patterns_for_binding(b)
                                .iter()
                                .find_map(|p| source_map.pat_syntax(*p).ok())
                            {
                                Some(s) => s.map(|it| it.into()),
                                None => continue,
                            }
                        }
                        mir::MirSpan::Unknown => continue,
                    };
                    acc.push(
                        MovedOutOfRef { ty: Type::new_for_crate(krate, moof.ty.clone()), span }
                            .into(),
                    )
                }
                let mol = &borrowck_result.mutability_of_locals;
                for (binding_id, binding_data) in body.bindings.iter() {
                    if binding_data.problems.is_some() {
                        // We should report specific diagnostics for these problems, not `need-mut` and `unused-mut`.
                        continue;
                    }
                    let Some(&local) = mir_body.binding_locals.get(binding_id) else {
                        continue;
                    };
                    if source_map
                        .patterns_for_binding(binding_id)
                        .iter()
                        .any(|&pat| source_map.pat_syntax(pat).is_err())
                    {
                        // Skip synthetic bindings
                        continue;
                    }
                    let mut need_mut = &mol[local];
                    if body[binding_id].name == sym::self_
                        && need_mut == &mir::MutabilityReason::Unused
                    {
                        need_mut = &mir::MutabilityReason::Not;
                    }
                    let local = Local { parent: self.into(), binding_id };
                    let is_mut = body[binding_id].mode == BindingAnnotation::Mutable;

                    match (need_mut, is_mut) {
                        (mir::MutabilityReason::Unused, _) => {
                            let should_ignore = body[binding_id].name.as_str().starts_with('_');
                            if !should_ignore {
                                acc.push(UnusedVariable { local }.into())
                            }
                        }
                        (mir::MutabilityReason::Mut { .. }, true)
                        | (mir::MutabilityReason::Not, false) => (),
                        (mir::MutabilityReason::Mut { spans }, false) => {
                            for span in spans {
                                let span: InFile<SyntaxNodePtr> = match span {
                                    mir::MirSpan::ExprId(e) => match source_map.expr_syntax(*e) {
                                        Ok(s) => s.map(|it| it.into()),
                                        Err(_) => continue,
                                    },
                                    mir::MirSpan::PatId(p) => match source_map.pat_syntax(*p) {
                                        Ok(s) => s.map(|it| it.into()),
                                        Err(_) => continue,
                                    },
                                    mir::MirSpan::BindingId(b) => {
                                        match source_map
                                            .patterns_for_binding(*b)
                                            .iter()
                                            .find_map(|p| source_map.pat_syntax(*p).ok())
                                        {
                                            Some(s) => s.map(|it| it.into()),
                                            None => continue,
                                        }
                                    }
                                    mir::MirSpan::SelfParam => match source_map.self_param_syntax()
                                    {
                                        Some(s) => s.map(|it| it.into()),
                                        None => continue,
                                    },
                                    mir::MirSpan::Unknown => continue,
                                };
                                acc.push(NeedMut { local, span }.into());
                            }
                        }
                        (mir::MutabilityReason::Not, true) => {
                            if !infer.mutated_bindings_in_closure.contains(&binding_id) {
                                let should_ignore = body[binding_id].name.as_str().starts_with('_');
                                if !should_ignore {
                                    acc.push(UnusedMut { local }.into())
                                }
                            }
                        }
                    }
                }
            }
        }

        for diagnostic in BodyValidationDiagnostic::collect(db, self.into(), style_lints) {
            acc.extend(AnyDiagnostic::body_validation_diagnostic(db, diagnostic, &source_map));
        }

        let def: ModuleDef = match self {
            DefWithBody::Function(it) => it.into(),
            DefWithBody::Static(it) => it.into(),
            DefWithBody::Const(it) => it.into(),
            DefWithBody::Variant(it) => it.into(),
        };
        for diag in hir_ty::diagnostics::incorrect_case(db, def.into()) {
            acc.push(diag.into())
        }
    }
}

fn expr_store_diagnostics(
    db: &dyn HirDatabase,
    acc: &mut Vec<AnyDiagnostic<'_>>,
    source_map: &ExpressionStoreSourceMap,
) {
    for diag in source_map.diagnostics() {
        acc.push(match diag {
            ExpressionStoreDiagnostics::InactiveCode { node, cfg, opts } => {
                InactiveCode { node: *node, cfg: cfg.clone(), opts: opts.clone() }.into()
            }
            ExpressionStoreDiagnostics::MacroError { node, err } => {
                let RenderedExpandError { message, error, kind } = err.render_to_string(db);

                let editioned_file_id = EditionedFileId::from_span(db, err.span().anchor.file_id);
                let precise_location = if editioned_file_id == node.file_id {
                    Some(
                        err.span().range
                            + db.ast_id_map(editioned_file_id.into())
                                .get_erased(err.span().anchor.ast_id)
                                .text_range()
                                .start(),
                    )
                } else {
                    None
                };
                MacroError {
                    node: (node).map(|it| it.into()),
                    precise_location,
                    message,
                    error,
                    kind,
                }
                .into()
            }
            ExpressionStoreDiagnostics::UnresolvedMacroCall { node, path } => UnresolvedMacroCall {
                macro_call: (*node).map(|ast_ptr| ast_ptr.into()),
                precise_location: None,
                path: path.clone(),
                is_bang: true,
            }
            .into(),
            ExpressionStoreDiagnostics::AwaitOutsideOfAsync { node, location } => {
                AwaitOutsideOfAsync { node: *node, location: location.clone() }.into()
            }
            ExpressionStoreDiagnostics::UnreachableLabel { node, name } => {
                UnreachableLabel { node: *node, name: name.clone() }.into()
            }
            ExpressionStoreDiagnostics::UndeclaredLabel { node, name } => {
                UndeclaredLabel { node: *node, name: name.clone() }.into()
            }
        });
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Function {
    pub(crate) id: FunctionId,
}

impl Function {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        self.id.module(db).into()
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        db.function_signature(self.id).name.clone()
    }

    pub fn ty(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::from_value_def(db, self.id)
    }

    pub fn fn_ptr_type(self, db: &dyn HirDatabase) -> Type<'_> {
        let resolver = self.id.resolver(db);
        let substs = TyBuilder::placeholder_subst(db, self.id);
        let callable_sig = db.callable_item_signature(self.id.into()).substitute(Interner, &substs);
        let ty = TyKind::Function(callable_sig.to_fn_ptr()).intern(Interner);
        Type::new_with_resolver_inner(db, &resolver, ty)
    }

    /// Get this function's return type
    pub fn ret_type(self, db: &dyn HirDatabase) -> Type<'_> {
        let resolver = self.id.resolver(db);
        let substs = TyBuilder::placeholder_subst(db, self.id);
        let callable_sig = db.callable_item_signature(self.id.into()).substitute(Interner, &substs);
        let ty = callable_sig.ret().clone();
        Type::new_with_resolver_inner(db, &resolver, ty)
    }

    // FIXME: Find better API to also handle const generics
    pub fn ret_type_with_args<'db>(
        self,
        db: &'db dyn HirDatabase,
        generics: impl Iterator<Item = Type<'db>>,
    ) -> Type<'db> {
        let resolver = self.id.resolver(db);
        let parent_id: Option<GenericDefId> = match self.id.lookup(db).container {
            ItemContainerId::ImplId(it) => Some(it.into()),
            ItemContainerId::TraitId(it) => Some(it.into()),
            ItemContainerId::ModuleId(_) | ItemContainerId::ExternBlockId(_) => None,
        };
        let mut generics = generics.map(|it| it.ty);
        let mut filler = |x: &_| match x {
            ParamKind::Type => {
                generics.next().unwrap_or_else(|| TyKind::Error.intern(Interner)).cast(Interner)
            }
            ParamKind::Const(ty) => unknown_const_as_generic(ty.clone()),
            ParamKind::Lifetime => error_lifetime().cast(Interner),
        };

        let parent_substs =
            parent_id.map(|id| TyBuilder::subst_for_def(db, id, None).fill(&mut filler).build());
        let substs = TyBuilder::subst_for_def(db, self.id, parent_substs).fill(&mut filler).build();

        let callable_sig = db.callable_item_signature(self.id.into()).substitute(Interner, &substs);
        let ty = callable_sig.ret().clone();
        Type::new_with_resolver_inner(db, &resolver, ty)
    }

    pub fn async_ret_type<'db>(self, db: &'db dyn HirDatabase) -> Option<Type<'db>> {
        if !self.is_async(db) {
            return None;
        }
        let resolver = self.id.resolver(db);
        let substs = TyBuilder::placeholder_subst(db, self.id);
        let callable_sig = db.callable_item_signature(self.id.into()).substitute(Interner, &substs);
        let ret_ty = callable_sig.ret().clone();
        for pred in ret_ty.impl_trait_bounds(db).into_iter().flatten() {
            if let WhereClause::AliasEq(output_eq) = pred.into_value_and_skipped_binders().0 {
                return Type::new_with_resolver_inner(db, &resolver, output_eq.ty).into();
            }
        }
        None
    }

    pub fn has_self_param(self, db: &dyn HirDatabase) -> bool {
        db.function_signature(self.id).has_self_param()
    }

    pub fn self_param(self, db: &dyn HirDatabase) -> Option<SelfParam> {
        self.has_self_param(db).then_some(SelfParam { func: self.id })
    }

    pub fn assoc_fn_params(self, db: &dyn HirDatabase) -> Vec<Param<'_>> {
        let environment = db.trait_environment(self.id.into());
        let substs = TyBuilder::placeholder_subst(db, self.id);
        let callable_sig = db.callable_item_signature(self.id.into()).substitute(Interner, &substs);
        callable_sig
            .params()
            .iter()
            .enumerate()
            .map(|(idx, ty)| {
                let ty = Type {
                    env: environment.clone(),
                    ty: ty.clone(),
                    _pd: PhantomCovariantLifetime::new(),
                };
                Param { func: Callee::Def(CallableDefId::FunctionId(self.id)), ty, idx }
            })
            .collect()
    }

    pub fn num_params(self, db: &dyn HirDatabase) -> usize {
        db.function_signature(self.id).params.len()
    }

    pub fn method_params(self, db: &dyn HirDatabase) -> Option<Vec<Param<'_>>> {
        self.self_param(db)?;
        Some(self.params_without_self(db))
    }

    pub fn params_without_self(self, db: &dyn HirDatabase) -> Vec<Param<'_>> {
        let environment = db.trait_environment(self.id.into());
        let substs = TyBuilder::placeholder_subst(db, self.id);
        let callable_sig = db.callable_item_signature(self.id.into()).substitute(Interner, &substs);
        let skip = if db.function_signature(self.id).has_self_param() { 1 } else { 0 };
        callable_sig
            .params()
            .iter()
            .enumerate()
            .skip(skip)
            .map(|(idx, ty)| {
                let ty = Type {
                    env: environment.clone(),
                    ty: ty.clone(),
                    _pd: PhantomCovariantLifetime::new(),
                };
                Param { func: Callee::Def(CallableDefId::FunctionId(self.id)), ty, idx }
            })
            .collect()
    }

    // FIXME: Find better API to also handle const generics
    pub fn params_without_self_with_args<'db>(
        self,
        db: &'db dyn HirDatabase,
        generics: impl Iterator<Item = Type<'db>>,
    ) -> Vec<Param<'db>> {
        let environment = db.trait_environment(self.id.into());
        let parent_id: Option<GenericDefId> = match self.id.lookup(db).container {
            ItemContainerId::ImplId(it) => Some(it.into()),
            ItemContainerId::TraitId(it) => Some(it.into()),
            ItemContainerId::ModuleId(_) | ItemContainerId::ExternBlockId(_) => None,
        };
        let mut generics = generics.map(|it| it.ty);
        let parent_substs = parent_id.map(|id| {
            TyBuilder::subst_for_def(db, id, None)
                .fill(|x| match x {
                    ParamKind::Type => generics
                        .next()
                        .unwrap_or_else(|| TyKind::Error.intern(Interner))
                        .cast(Interner),
                    ParamKind::Const(ty) => unknown_const_as_generic(ty.clone()),
                    ParamKind::Lifetime => error_lifetime().cast(Interner),
                })
                .build()
        });

        let substs = TyBuilder::subst_for_def(db, self.id, parent_substs)
            .fill(|_| {
                let ty = generics.next().unwrap_or_else(|| TyKind::Error.intern(Interner));
                GenericArg::new(Interner, GenericArgData::Ty(ty))
            })
            .build();
        let callable_sig = db.callable_item_signature(self.id.into()).substitute(Interner, &substs);
        let skip = if db.function_signature(self.id).has_self_param() { 1 } else { 0 };
        callable_sig
            .params()
            .iter()
            .enumerate()
            .skip(skip)
            .map(|(idx, ty)| {
                let ty = Type {
                    env: environment.clone(),
                    ty: ty.clone(),
                    _pd: PhantomCovariantLifetime::new(),
                };
                Param { func: Callee::Def(CallableDefId::FunctionId(self.id)), ty, idx }
            })
            .collect()
    }

    pub fn is_const(self, db: &dyn HirDatabase) -> bool {
        db.function_signature(self.id).is_const()
    }

    pub fn is_async(self, db: &dyn HirDatabase) -> bool {
        db.function_signature(self.id).is_async()
    }

    pub fn is_varargs(self, db: &dyn HirDatabase) -> bool {
        db.function_signature(self.id).is_varargs()
    }

    pub fn extern_block(self, db: &dyn HirDatabase) -> Option<ExternBlock> {
        match self.id.lookup(db).container {
            ItemContainerId::ExternBlockId(id) => Some(ExternBlock { id }),
            _ => None,
        }
    }

    pub fn returns_impl_future(self, db: &dyn HirDatabase) -> bool {
        if self.is_async(db) {
            return true;
        }

        let ret_type = self.ret_type(db);
        let Some(impl_traits) = ret_type.as_impl_traits(db) else { return false };
        let Some(future_trait_id) = LangItem::Future.resolve_trait(db, self.ty(db).env.krate)
        else {
            return false;
        };
        let Some(sized_trait_id) = LangItem::Sized.resolve_trait(db, self.ty(db).env.krate) else {
            return false;
        };

        let mut has_impl_future = false;
        impl_traits
            .filter(|t| {
                let fut = t.id == future_trait_id;
                has_impl_future |= fut;
                !fut && t.id != sized_trait_id
            })
            // all traits but the future trait must be auto traits
            .all(|t| t.is_auto(db))
            && has_impl_future
    }

    /// Does this function have `#[test]` attribute?
    pub fn is_test(self, db: &dyn HirDatabase) -> bool {
        db.attrs(self.id.into()).is_test()
    }

    /// is this a `fn main` or a function with an `export_name` of `main`?
    pub fn is_main(self, db: &dyn HirDatabase) -> bool {
        db.attrs(self.id.into()).export_name() == Some(&sym::main)
            || self.module(db).is_crate_root() && db.function_signature(self.id).name == sym::main
    }

    /// Is this a function with an `export_name` of `main`?
    pub fn exported_main(self, db: &dyn HirDatabase) -> bool {
        db.attrs(self.id.into()).export_name() == Some(&sym::main)
    }

    /// Does this function have the ignore attribute?
    pub fn is_ignore(self, db: &dyn HirDatabase) -> bool {
        db.attrs(self.id.into()).is_ignore()
    }

    /// Does this function have `#[bench]` attribute?
    pub fn is_bench(self, db: &dyn HirDatabase) -> bool {
        db.attrs(self.id.into()).is_bench()
    }

    /// Is this function marked as unstable with `#[feature]` attribute?
    pub fn is_unstable(self, db: &dyn HirDatabase) -> bool {
        db.attrs(self.id.into()).is_unstable()
    }

    pub fn is_unsafe_to_call(
        self,
        db: &dyn HirDatabase,
        caller: Option<Function>,
        call_edition: Edition,
    ) -> bool {
        let target_features = caller
            .map(|caller| hir_ty::TargetFeatures::from_attrs(&db.attrs(caller.id.into())))
            .unwrap_or_default();
        matches!(
            hir_ty::is_fn_unsafe_to_call(db, self.id, &target_features, call_edition),
            hir_ty::Unsafety::Unsafe
        )
    }

    /// Whether this function declaration has a definition.
    ///
    /// This is false in the case of required (not provided) trait methods.
    pub fn has_body(self, db: &dyn HirDatabase) -> bool {
        db.function_signature(self.id).has_body()
    }

    pub fn as_proc_macro(self, db: &dyn HirDatabase) -> Option<Macro> {
        let attrs = db.attrs(self.id.into());
        // FIXME: Store this in FunctionData flags?
        if !(attrs.is_proc_macro()
            || attrs.is_proc_macro_attribute()
            || attrs.is_proc_macro_derive())
        {
            return None;
        }
        let def_map = crate_def_map(db, HasModule::krate(&self.id, db));
        def_map.fn_as_proc_macro(self.id).map(|id| Macro { id: id.into() })
    }

    pub fn eval(
        self,
        db: &dyn HirDatabase,
        span_formatter: impl Fn(FileId, TextRange) -> String,
    ) -> Result<String, ConstEvalError> {
        let body = db.monomorphized_mir_body(
            self.id.into(),
            Substitution::empty(Interner),
            db.trait_environment(self.id.into()),
        )?;
        let (result, output) = interpret_mir(db, body, false, None)?;
        let mut text = match result {
            Ok(_) => "pass".to_owned(),
            Err(e) => {
                let mut r = String::new();
                _ = e.pretty_print(
                    &mut r,
                    db,
                    &span_formatter,
                    self.krate(db).to_display_target(db),
                );
                r
            }
        };
        let stdout = output.stdout().into_owned();
        if !stdout.is_empty() {
            text += "\n--------- stdout ---------\n";
            text += &stdout;
        }
        let stderr = output.stdout().into_owned();
        if !stderr.is_empty() {
            text += "\n--------- stderr ---------\n";
            text += &stderr;
        }
        Ok(text)
    }
}

// Note: logically, this belongs to `hir_ty`, but we are not using it there yet.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Shared,
    Exclusive,
    Owned,
}

impl From<hir_ty::Mutability> for Access {
    fn from(mutability: hir_ty::Mutability) -> Access {
        match mutability {
            hir_ty::Mutability::Not => Access::Shared,
            hir_ty::Mutability::Mut => Access::Exclusive,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Param<'db> {
    func: Callee,
    /// The index in parameter list, including self parameter.
    idx: usize,
    ty: Type<'db>,
}

impl<'db> Param<'db> {
    pub fn parent_fn(&self) -> Option<Function> {
        match self.func {
            Callee::Def(CallableDefId::FunctionId(f)) => Some(f.into()),
            _ => None,
        }
    }

    // pub fn parent_closure(&self) -> Option<Closure> {
    //     self.func.as_ref().right().cloned()
    // }

    pub fn index(&self) -> usize {
        self.idx
    }

    pub fn ty(&self) -> &Type<'db> {
        &self.ty
    }

    pub fn name(&self, db: &dyn HirDatabase) -> Option<Name> {
        Some(self.as_local(db)?.name(db))
    }

    pub fn as_local(&self, db: &dyn HirDatabase) -> Option<Local> {
        match self.func {
            Callee::Def(CallableDefId::FunctionId(it)) => {
                let parent = DefWithBodyId::FunctionId(it);
                let body = db.body(parent);
                if let Some(self_param) = body.self_param.filter(|_| self.idx == 0) {
                    Some(Local { parent, binding_id: self_param })
                } else if let Pat::Bind { id, .. } =
                    &body[body.params[self.idx - body.self_param.is_some() as usize]]
                {
                    Some(Local { parent, binding_id: *id })
                } else {
                    None
                }
            }
            Callee::Closure(closure, _) => {
                let c = db.lookup_intern_closure(closure.into());
                let body = db.body(c.0);
                if let Expr::Closure { args, .. } = &body[c.1] {
                    if let Pat::Bind { id, .. } = &body[args[self.idx]] {
                        return Some(Local { parent: c.0, binding_id: *id });
                    }
                }
                None
            }
            _ => None,
        }
    }

    pub fn pattern_source(self, db: &dyn HirDatabase) -> Option<ast::Pat> {
        self.source(db).and_then(|p| p.value.right()?.pat())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SelfParam {
    func: FunctionId,
}

impl SelfParam {
    pub fn access(self, db: &dyn HirDatabase) -> Access {
        let func_data = db.function_signature(self.func);
        func_data
            .params
            .first()
            .map(|&param| match &func_data.store[param] {
                TypeRef::Reference(ref_) => match ref_.mutability {
                    hir_def::type_ref::Mutability::Shared => Access::Shared,
                    hir_def::type_ref::Mutability::Mut => Access::Exclusive,
                },
                _ => Access::Owned,
            })
            .unwrap_or(Access::Owned)
    }

    pub fn parent_fn(&self) -> Function {
        Function::from(self.func)
    }

    pub fn ty<'db>(&self, db: &'db dyn HirDatabase) -> Type<'db> {
        let substs = TyBuilder::placeholder_subst(db, self.func);
        let callable_sig =
            db.callable_item_signature(self.func.into()).substitute(Interner, &substs);
        let environment = db.trait_environment(self.func.into());
        let ty = callable_sig.params()[0].clone();
        Type { env: environment, ty, _pd: PhantomCovariantLifetime::new() }
    }

    // FIXME: Find better API to also handle const generics
    pub fn ty_with_args<'db>(
        &self,
        db: &'db dyn HirDatabase,
        generics: impl Iterator<Item = Type<'db>>,
    ) -> Type<'db> {
        let parent_id: GenericDefId = match self.func.lookup(db).container {
            ItemContainerId::ImplId(it) => it.into(),
            ItemContainerId::TraitId(it) => it.into(),
            ItemContainerId::ModuleId(_) | ItemContainerId::ExternBlockId(_) => {
                panic!("Never get here")
            }
        };

        let mut generics = generics.map(|it| it.ty);
        let mut filler = |x: &_| match x {
            ParamKind::Type => {
                generics.next().unwrap_or_else(|| TyKind::Error.intern(Interner)).cast(Interner)
            }
            ParamKind::Const(ty) => unknown_const_as_generic(ty.clone()),
            ParamKind::Lifetime => error_lifetime().cast(Interner),
        };

        let parent_substs = TyBuilder::subst_for_def(db, parent_id, None).fill(&mut filler).build();
        let substs =
            TyBuilder::subst_for_def(db, self.func, Some(parent_substs)).fill(&mut filler).build();
        let callable_sig =
            db.callable_item_signature(self.func.into()).substitute(Interner, &substs);
        let environment = db.trait_environment(self.func.into());
        let ty = callable_sig.params()[0].clone();
        Type { env: environment, ty, _pd: PhantomCovariantLifetime::new() }
    }
}

impl HasVisibility for Function {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        db.assoc_visibility(self.id.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExternCrateDecl {
    pub(crate) id: ExternCrateId,
}

impl ExternCrateDecl {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        self.id.module(db).into()
    }

    pub fn resolved_crate(self, db: &dyn HirDatabase) -> Option<Crate> {
        let loc = self.id.lookup(db);
        let krate = loc.container.krate();
        let name = self.name(db);
        if name == sym::self_ {
            Some(krate.into())
        } else {
            krate.data(db).dependencies.iter().find_map(|dep| {
                if dep.name.symbol() == name.symbol() { Some(dep.crate_id.into()) } else { None }
            })
        }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        let loc = self.id.lookup(db);
        let source = loc.source(db);
        as_name_opt(source.value.name_ref())
    }

    pub fn alias(self, db: &dyn HirDatabase) -> Option<ImportAlias> {
        let loc = self.id.lookup(db);
        let source = loc.source(db);
        let rename = source.value.rename()?;
        if let Some(name) = rename.name() {
            Some(ImportAlias::Alias(name.as_name()))
        } else if rename.underscore_token().is_some() {
            Some(ImportAlias::Underscore)
        } else {
            None
        }
    }

    /// Returns the name under which this crate is made accessible, taking `_` into account.
    pub fn alias_or_name(self, db: &dyn HirDatabase) -> Option<Name> {
        match self.alias(db) {
            Some(ImportAlias::Underscore) => None,
            Some(ImportAlias::Alias(alias)) => Some(alias),
            None => Some(self.name(db)),
        }
    }
}

impl HasVisibility for ExternCrateDecl {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        let loc = self.id.lookup(db);
        let source = loc.source(db);
        visibility_from_ast(db, self.id, source.map(|src| src.visibility()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Const {
    pub(crate) id: ConstId,
}

impl Const {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        Module { id: self.id.module(db) }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Option<Name> {
        db.const_signature(self.id).name.clone()
    }

    pub fn value(self, db: &dyn HirDatabase) -> Option<ast::Expr> {
        self.source(db)?.value.body()
    }

    pub fn ty(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::from_value_def(db, self.id)
    }

    /// Evaluate the constant.
    pub fn eval(self, db: &dyn HirDatabase) -> Result<EvaluatedConst, ConstEvalError> {
        db.const_eval(self.id.into(), Substitution::empty(Interner), None)
            .map(|it| EvaluatedConst { const_: it, def: self.id.into() })
    }
}

impl HasVisibility for Const {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        db.assoc_visibility(self.id.into())
    }
}

pub struct EvaluatedConst {
    def: DefWithBodyId,
    const_: hir_ty::Const,
}

impl EvaluatedConst {
    pub fn render(&self, db: &dyn HirDatabase, display_target: DisplayTarget) -> String {
        format!("{}", self.const_.display(db, display_target))
    }

    pub fn render_debug(&self, db: &dyn HirDatabase) -> Result<String, MirEvalError> {
        let data = self.const_.data(Interner);
        if let TyKind::Scalar(s) = data.ty.kind(Interner) {
            if matches!(s, Scalar::Int(_) | Scalar::Uint(_)) {
                if let hir_ty::ConstValue::Concrete(c) = &data.value {
                    if let hir_ty::ConstScalar::Bytes(b, _) = &c.interned {
                        let value = u128::from_le_bytes(mir::pad16(b, false));
                        let value_signed =
                            i128::from_le_bytes(mir::pad16(b, matches!(s, Scalar::Int(_))));
                        let mut result = if let Scalar::Int(_) = s {
                            value_signed.to_string()
                        } else {
                            value.to_string()
                        };
                        if value >= 10 {
                            format_to!(result, " ({value:#X})");
                            return Ok(result);
                        } else {
                            return Ok(result);
                        }
                    }
                }
            }
        }
        mir::render_const_using_debug_impl(db, self.def, &self.const_)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Static {
    pub(crate) id: StaticId,
}

impl Static {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        Module { id: self.id.module(db) }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        db.static_signature(self.id).name.clone()
    }

    pub fn is_mut(self, db: &dyn HirDatabase) -> bool {
        db.static_signature(self.id).flags.contains(StaticFlags::MUTABLE)
    }

    pub fn value(self, db: &dyn HirDatabase) -> Option<ast::Expr> {
        self.source(db)?.value.body()
    }

    pub fn ty(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::from_value_def(db, self.id)
    }

    pub fn extern_block(self, db: &dyn HirDatabase) -> Option<ExternBlock> {
        match self.id.lookup(db).container {
            ItemContainerId::ExternBlockId(id) => Some(ExternBlock { id }),
            _ => None,
        }
    }

    /// Evaluate the static initializer.
    pub fn eval(self, db: &dyn HirDatabase) -> Result<EvaluatedConst, ConstEvalError> {
        db.const_eval(self.id.into(), Substitution::empty(Interner), None)
            .map(|it| EvaluatedConst { const_: it, def: self.id.into() })
    }
}

impl HasVisibility for Static {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        let loc = self.id.lookup(db);
        let source = loc.source(db);
        visibility_from_ast(db, self.id, source.map(|src| src.visibility()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Trait {
    pub(crate) id: TraitId,
}

impl Trait {
    pub fn lang(db: &dyn HirDatabase, krate: Crate, name: &Name) -> Option<Trait> {
        LangItem::from_name(name)?.resolve_trait(db, krate.into()).map(Into::into)
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        Module { id: self.id.lookup(db).container }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        db.trait_signature(self.id).name.clone()
    }

    pub fn direct_supertraits(self, db: &dyn HirDatabase) -> Vec<Trait> {
        let traits = direct_super_traits(db, self.into());
        traits.iter().map(|tr| Trait::from(*tr)).collect()
    }

    pub fn all_supertraits(self, db: &dyn HirDatabase) -> Vec<Trait> {
        let traits = all_super_traits(db, self.into());
        traits.iter().map(|tr| Trait::from(*tr)).collect()
    }

    pub fn function(self, db: &dyn HirDatabase, name: impl PartialEq<Name>) -> Option<Function> {
        self.id.trait_items(db).items.iter().find(|(n, _)| name == *n).and_then(|&(_, it)| match it
        {
            AssocItemId::FunctionId(id) => Some(Function { id }),
            _ => None,
        })
    }

    pub fn items(self, db: &dyn HirDatabase) -> Vec<AssocItem> {
        self.id.trait_items(db).items.iter().map(|(_name, it)| (*it).into()).collect()
    }

    pub fn items_with_supertraits(self, db: &dyn HirDatabase) -> Vec<AssocItem> {
        self.all_supertraits(db).into_iter().flat_map(|tr| tr.items(db)).collect()
    }

    pub fn is_auto(self, db: &dyn HirDatabase) -> bool {
        db.trait_signature(self.id).flags.contains(TraitFlags::AUTO)
    }

    pub fn is_unsafe(&self, db: &dyn HirDatabase) -> bool {
        db.trait_signature(self.id).flags.contains(TraitFlags::UNSAFE)
    }

    pub fn type_or_const_param_count(
        &self,
        db: &dyn HirDatabase,
        count_required_only: bool,
    ) -> usize {
        db.generic_params(self.id.into())
            .iter_type_or_consts()
            .filter(|(_, ty)| !matches!(ty, TypeOrConstParamData::TypeParamData(ty) if ty.provenance != TypeParamProvenance::TypeParamList))
            .filter(|(_, ty)| !count_required_only || !ty.has_default())
            .count()
    }

    pub fn dyn_compatibility(&self, db: &dyn HirDatabase) -> Option<DynCompatibilityViolation> {
        hir_ty::dyn_compatibility::dyn_compatibility(db, self.id)
    }

    pub fn dyn_compatibility_all_violations(
        &self,
        db: &dyn HirDatabase,
    ) -> Option<Vec<DynCompatibilityViolation>> {
        let mut violations = vec![];
        _ = hir_ty::dyn_compatibility::dyn_compatibility_with_callback(
            db,
            self.id,
            &mut |violation| {
                violations.push(violation);
                ControlFlow::Continue(())
            },
        );
        violations.is_empty().not().then_some(violations)
    }

    fn all_macro_calls(&self, db: &dyn HirDatabase) -> Box<[(AstId<ast::Item>, MacroCallId)]> {
        self.id.trait_items(db).macro_calls.to_vec().into_boxed_slice()
    }

    /// `#[rust_analyzer::completions(...)]` mode.
    pub fn complete(self, db: &dyn HirDatabase) -> Complete {
        Complete::extract(true, &self.attrs(db))
    }
}

impl HasVisibility for Trait {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        let loc = self.id.lookup(db);
        let source = loc.source(db);
        visibility_from_ast(db, self.id, source.map(|src| src.visibility()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraitAlias {
    pub(crate) id: TraitAliasId,
}

impl TraitAlias {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        Module { id: self.id.lookup(db).container }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        db.trait_alias_signature(self.id).name.clone()
    }
}

impl HasVisibility for TraitAlias {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        let loc = self.id.lookup(db);
        let source = loc.source(db);
        visibility_from_ast(db, self.id, source.map(|src| src.visibility()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeAlias {
    pub(crate) id: TypeAliasId,
}

impl TypeAlias {
    pub fn has_non_default_type_params(self, db: &dyn HirDatabase) -> bool {
        let subst = db.generic_defaults(self.id.into());
        (subst.is_empty() && db.generic_params(self.id.into()).len_type_or_consts() != 0)
            || subst.iter().any(|ty| match ty.skip_binders().data(Interner) {
                GenericArgData::Ty(it) => it.is_unknown(),
                _ => false,
            })
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        Module { id: self.id.module(db) }
    }

    pub fn ty(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::from_def(db, self.id)
    }

    pub fn ty_placeholders(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::from_def_placeholders(db, self.id)
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        db.type_alias_signature(self.id).name.clone()
    }
}

impl HasVisibility for TypeAlias {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        db.assoc_visibility(self.id.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExternBlock {
    pub(crate) id: ExternBlockId,
}

impl ExternBlock {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        Module { id: self.id.module(db) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticLifetime;

impl StaticLifetime {
    pub fn name(self) -> Name {
        Name::new_symbol_root(sym::tick_static)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuiltinType {
    pub(crate) inner: hir_def::builtin_type::BuiltinType,
}

impl BuiltinType {
    // Constructors are added on demand, feel free to add more.
    pub fn str() -> BuiltinType {
        BuiltinType { inner: hir_def::builtin_type::BuiltinType::Str }
    }

    pub fn i32() -> BuiltinType {
        BuiltinType {
            inner: hir_def::builtin_type::BuiltinType::Int(hir_ty::primitive::BuiltinInt::I32),
        }
    }

    pub fn ty<'db>(self, db: &'db dyn HirDatabase) -> Type<'db> {
        let core = Crate::core(db).map(|core| core.id).unwrap_or_else(|| db.all_crates()[0]);
        Type::new_for_crate(core, TyBuilder::builtin(self.inner))
    }

    pub fn name(self) -> Name {
        self.inner.as_name()
    }

    pub fn is_int(&self) -> bool {
        matches!(self.inner, hir_def::builtin_type::BuiltinType::Int(_))
    }

    pub fn is_uint(&self) -> bool {
        matches!(self.inner, hir_def::builtin_type::BuiltinType::Uint(_))
    }

    pub fn is_float(&self) -> bool {
        matches!(self.inner, hir_def::builtin_type::BuiltinType::Float(_))
    }

    pub fn is_f16(&self) -> bool {
        matches!(
            self.inner,
            hir_def::builtin_type::BuiltinType::Float(hir_def::builtin_type::BuiltinFloat::F16)
        )
    }

    pub fn is_f32(&self) -> bool {
        matches!(
            self.inner,
            hir_def::builtin_type::BuiltinType::Float(hir_def::builtin_type::BuiltinFloat::F32)
        )
    }

    pub fn is_f64(&self) -> bool {
        matches!(
            self.inner,
            hir_def::builtin_type::BuiltinType::Float(hir_def::builtin_type::BuiltinFloat::F64)
        )
    }

    pub fn is_f128(&self) -> bool {
        matches!(
            self.inner,
            hir_def::builtin_type::BuiltinType::Float(hir_def::builtin_type::BuiltinFloat::F128)
        )
    }

    pub fn is_char(&self) -> bool {
        matches!(self.inner, hir_def::builtin_type::BuiltinType::Char)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self.inner, hir_def::builtin_type::BuiltinType::Bool)
    }

    pub fn is_str(&self) -> bool {
        matches!(self.inner, hir_def::builtin_type::BuiltinType::Str)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Macro {
    pub(crate) id: MacroId,
}

impl Macro {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        Module { id: self.id.module(db) }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        match self.id {
            MacroId::Macro2Id(id) => {
                let loc = id.lookup(db);
                let source = loc.source(db);
                as_name_opt(source.value.name())
            }
            MacroId::MacroRulesId(id) => {
                let loc = id.lookup(db);
                let source = loc.source(db);
                as_name_opt(source.value.name())
            }
            MacroId::ProcMacroId(id) => {
                let loc = id.lookup(db);
                let source = loc.source(db);
                match loc.kind {
                    ProcMacroKind::CustomDerive => db
                        .attrs(id.into())
                        .parse_proc_macro_derive()
                        .map_or_else(|| as_name_opt(source.value.name()), |(it, _)| it),
                    ProcMacroKind::Bang | ProcMacroKind::Attr => as_name_opt(source.value.name()),
                }
            }
        }
    }

    pub fn is_macro_export(self, db: &dyn HirDatabase) -> bool {
        matches!(self.id, MacroId::MacroRulesId(_) if db.attrs(self.id.into()).by_key(sym::macro_export).exists())
    }

    pub fn is_proc_macro(self) -> bool {
        matches!(self.id, MacroId::ProcMacroId(_))
    }

    pub fn kind(&self, db: &dyn HirDatabase) -> MacroKind {
        match self.id {
            MacroId::Macro2Id(it) => match it.lookup(db).expander {
                MacroExpander::Declarative => MacroKind::Declarative,
                MacroExpander::BuiltIn(_) | MacroExpander::BuiltInEager(_) => {
                    MacroKind::DeclarativeBuiltIn
                }
                MacroExpander::BuiltInAttr(_) => MacroKind::AttrBuiltIn,
                MacroExpander::BuiltInDerive(_) => MacroKind::DeriveBuiltIn,
            },
            MacroId::MacroRulesId(it) => match it.lookup(db).expander {
                MacroExpander::Declarative => MacroKind::Declarative,
                MacroExpander::BuiltIn(_) | MacroExpander::BuiltInEager(_) => {
                    MacroKind::DeclarativeBuiltIn
                }
                MacroExpander::BuiltInAttr(_) => MacroKind::AttrBuiltIn,
                MacroExpander::BuiltInDerive(_) => MacroKind::DeriveBuiltIn,
            },
            MacroId::ProcMacroId(it) => match it.lookup(db).kind {
                ProcMacroKind::CustomDerive => MacroKind::Derive,
                ProcMacroKind::Bang => MacroKind::ProcMacro,
                ProcMacroKind::Attr => MacroKind::Attr,
            },
        }
    }

    pub fn is_fn_like(&self, db: &dyn HirDatabase) -> bool {
        matches!(
            self.kind(db),
            MacroKind::Declarative | MacroKind::DeclarativeBuiltIn | MacroKind::ProcMacro
        )
    }

    pub fn is_builtin_derive(&self, db: &dyn HirDatabase) -> bool {
        match self.id {
            MacroId::Macro2Id(it) => {
                matches!(it.lookup(db).expander, MacroExpander::BuiltInDerive(_))
            }
            MacroId::MacroRulesId(it) => {
                matches!(it.lookup(db).expander, MacroExpander::BuiltInDerive(_))
            }
            MacroId::ProcMacroId(_) => false,
        }
    }

    pub fn is_env_or_option_env(&self, db: &dyn HirDatabase) -> bool {
        match self.id {
            MacroId::Macro2Id(it) => {
                matches!(it.lookup(db).expander, MacroExpander::BuiltInEager(eager) if eager.is_env_or_option_env())
            }
            MacroId::MacroRulesId(it) => {
                matches!(it.lookup(db).expander, MacroExpander::BuiltInEager(eager) if eager.is_env_or_option_env())
            }
            MacroId::ProcMacroId(_) => false,
        }
    }

    pub fn is_asm_or_global_asm(&self, db: &dyn HirDatabase) -> bool {
        match self.id {
            MacroId::Macro2Id(it) => {
                matches!(it.lookup(db).expander, MacroExpander::BuiltIn(m) if m.is_asm())
            }
            MacroId::MacroRulesId(it) => {
                matches!(it.lookup(db).expander, MacroExpander::BuiltIn(m) if m.is_asm())
            }
            MacroId::ProcMacroId(_) => false,
        }
    }

    pub fn is_attr(&self, db: &dyn HirDatabase) -> bool {
        matches!(self.kind(db), MacroKind::Attr | MacroKind::AttrBuiltIn)
    }

    pub fn is_derive(&self, db: &dyn HirDatabase) -> bool {
        matches!(self.kind(db), MacroKind::Derive | MacroKind::DeriveBuiltIn)
    }
}

impl HasVisibility for Macro {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        match self.id {
            MacroId::Macro2Id(id) => {
                let loc = id.lookup(db);
                let source = loc.source(db);
                visibility_from_ast(db, id, source.map(|src| src.visibility()))
            }
            MacroId::MacroRulesId(_) => Visibility::Public,
            MacroId::ProcMacroId(_) => Visibility::Public,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ItemInNs {
    Types(ModuleDef),
    Values(ModuleDef),
    Macros(Macro),
}

impl From<Macro> for ItemInNs {
    fn from(it: Macro) -> Self {
        Self::Macros(it)
    }
}

impl From<ModuleDef> for ItemInNs {
    fn from(module_def: ModuleDef) -> Self {
        match module_def {
            ModuleDef::Static(_) | ModuleDef::Const(_) | ModuleDef::Function(_) => {
                ItemInNs::Values(module_def)
            }
            ModuleDef::Macro(it) => ItemInNs::Macros(it),
            _ => ItemInNs::Types(module_def),
        }
    }
}

impl ItemInNs {
    pub fn into_module_def(self) -> ModuleDef {
        match self {
            ItemInNs::Types(id) | ItemInNs::Values(id) => id,
            ItemInNs::Macros(id) => ModuleDef::Macro(id),
        }
    }

    /// Returns the crate defining this item (or `None` if `self` is built-in).
    pub fn krate(&self, db: &dyn HirDatabase) -> Option<Crate> {
        match self {
            ItemInNs::Types(did) | ItemInNs::Values(did) => did.module(db).map(|m| m.krate()),
            ItemInNs::Macros(id) => Some(id.module(db).krate()),
        }
    }

    pub fn attrs(&self, db: &dyn HirDatabase) -> Option<AttrsWithOwner> {
        match self {
            ItemInNs::Types(it) | ItemInNs::Values(it) => it.attrs(db),
            ItemInNs::Macros(it) => Some(it.attrs(db)),
        }
    }
}

/// Invariant: `inner.as_extern_assoc_item(db).is_some()`
/// We do not actively enforce this invariant.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ExternAssocItem {
    Function(Function),
    Static(Static),
    TypeAlias(TypeAlias),
}

pub trait AsExternAssocItem {
    fn as_extern_assoc_item(self, db: &dyn HirDatabase) -> Option<ExternAssocItem>;
}

impl AsExternAssocItem for Function {
    fn as_extern_assoc_item(self, db: &dyn HirDatabase) -> Option<ExternAssocItem> {
        as_extern_assoc_item(db, ExternAssocItem::Function, self.id)
    }
}

impl AsExternAssocItem for Static {
    fn as_extern_assoc_item(self, db: &dyn HirDatabase) -> Option<ExternAssocItem> {
        as_extern_assoc_item(db, ExternAssocItem::Static, self.id)
    }
}

impl AsExternAssocItem for TypeAlias {
    fn as_extern_assoc_item(self, db: &dyn HirDatabase) -> Option<ExternAssocItem> {
        as_extern_assoc_item(db, ExternAssocItem::TypeAlias, self.id)
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

#[derive(Debug, Clone)]
pub enum AssocItemContainer {
    Trait(Trait),
    Impl(Impl),
}

pub trait AsAssocItem {
    fn as_assoc_item(self, db: &dyn HirDatabase) -> Option<AssocItem>;
}

impl AsAssocItem for Function {
    fn as_assoc_item(self, db: &dyn HirDatabase) -> Option<AssocItem> {
        as_assoc_item(db, AssocItem::Function, self.id)
    }
}

impl AsAssocItem for Const {
    fn as_assoc_item(self, db: &dyn HirDatabase) -> Option<AssocItem> {
        as_assoc_item(db, AssocItem::Const, self.id)
    }
}

impl AsAssocItem for TypeAlias {
    fn as_assoc_item(self, db: &dyn HirDatabase) -> Option<AssocItem> {
        as_assoc_item(db, AssocItem::TypeAlias, self.id)
    }
}

impl AsAssocItem for ModuleDef {
    fn as_assoc_item(self, db: &dyn HirDatabase) -> Option<AssocItem> {
        match self {
            ModuleDef::Function(it) => it.as_assoc_item(db),
            ModuleDef::Const(it) => it.as_assoc_item(db),
            ModuleDef::TypeAlias(it) => it.as_assoc_item(db),
            _ => None,
        }
    }
}

impl AsAssocItem for DefWithBody {
    fn as_assoc_item(self, db: &dyn HirDatabase) -> Option<AssocItem> {
        match self {
            DefWithBody::Function(it) => it.as_assoc_item(db),
            DefWithBody::Const(it) => it.as_assoc_item(db),
            DefWithBody::Static(_) | DefWithBody::Variant(_) => None,
        }
    }
}

fn as_assoc_item<'db, ID, DEF, LOC>(
    db: &(dyn HirDatabase + 'db),
    ctor: impl FnOnce(DEF) -> AssocItem,
    id: ID,
) -> Option<AssocItem>
where
    ID: Lookup<Database = dyn DefDatabase, Data = AssocItemLoc<LOC>>,
    DEF: From<ID>,
    LOC: AstIdNode,
{
    match id.lookup(db).container {
        ItemContainerId::TraitId(_) | ItemContainerId::ImplId(_) => Some(ctor(DEF::from(id))),
        ItemContainerId::ModuleId(_) | ItemContainerId::ExternBlockId(_) => None,
    }
}

fn as_extern_assoc_item<'db, ID, DEF, LOC>(
    db: &(dyn HirDatabase + 'db),
    ctor: impl FnOnce(DEF) -> ExternAssocItem,
    id: ID,
) -> Option<ExternAssocItem>
where
    ID: Lookup<Database = dyn DefDatabase, Data = AssocItemLoc<LOC>>,
    DEF: From<ID>,
    LOC: AstIdNode,
{
    match id.lookup(db).container {
        ItemContainerId::ExternBlockId(_) => Some(ctor(DEF::from(id))),
        ItemContainerId::TraitId(_) | ItemContainerId::ImplId(_) | ItemContainerId::ModuleId(_) => {
            None
        }
    }
}

impl ExternAssocItem {
    pub fn name(self, db: &dyn HirDatabase) -> Name {
        match self {
            Self::Function(it) => it.name(db),
            Self::Static(it) => it.name(db),
            Self::TypeAlias(it) => it.name(db),
        }
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        match self {
            Self::Function(f) => f.module(db),
            Self::Static(c) => c.module(db),
            Self::TypeAlias(t) => t.module(db),
        }
    }

    pub fn as_function(self) -> Option<Function> {
        match self {
            Self::Function(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_static(self) -> Option<Static> {
        match self {
            Self::Static(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_type_alias(self) -> Option<TypeAlias> {
        match self {
            Self::TypeAlias(v) => Some(v),
            _ => None,
        }
    }
}

impl AssocItem {
    pub fn name(self, db: &dyn HirDatabase) -> Option<Name> {
        match self {
            AssocItem::Function(it) => Some(it.name(db)),
            AssocItem::Const(it) => it.name(db),
            AssocItem::TypeAlias(it) => Some(it.name(db)),
        }
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        match self {
            AssocItem::Function(f) => f.module(db),
            AssocItem::Const(c) => c.module(db),
            AssocItem::TypeAlias(t) => t.module(db),
        }
    }

    pub fn container(self, db: &dyn HirDatabase) -> AssocItemContainer {
        let container = match self {
            AssocItem::Function(it) => it.id.lookup(db).container,
            AssocItem::Const(it) => it.id.lookup(db).container,
            AssocItem::TypeAlias(it) => it.id.lookup(db).container,
        };
        match container {
            ItemContainerId::TraitId(id) => AssocItemContainer::Trait(id.into()),
            ItemContainerId::ImplId(id) => AssocItemContainer::Impl(id.into()),
            ItemContainerId::ModuleId(_) | ItemContainerId::ExternBlockId(_) => {
                panic!("invalid AssocItem")
            }
        }
    }

    pub fn container_trait(self, db: &dyn HirDatabase) -> Option<Trait> {
        match self.container(db) {
            AssocItemContainer::Trait(t) => Some(t),
            _ => None,
        }
    }

    pub fn implemented_trait(self, db: &dyn HirDatabase) -> Option<Trait> {
        match self.container(db) {
            AssocItemContainer::Impl(i) => i.trait_(db),
            _ => None,
        }
    }

    pub fn container_or_implemented_trait(self, db: &dyn HirDatabase) -> Option<Trait> {
        match self.container(db) {
            AssocItemContainer::Trait(t) => Some(t),
            AssocItemContainer::Impl(i) => i.trait_(db),
        }
    }

    pub fn implementing_ty(self, db: &dyn HirDatabase) -> Option<Type<'_>> {
        match self.container(db) {
            AssocItemContainer::Impl(i) => Some(i.self_ty(db)),
            _ => None,
        }
    }

    pub fn as_function(self) -> Option<Function> {
        match self {
            Self::Function(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_const(self) -> Option<Const> {
        match self {
            Self::Const(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_type_alias(self) -> Option<TypeAlias> {
        match self {
            Self::TypeAlias(v) => Some(v),
            _ => None,
        }
    }

    pub fn diagnostics<'db>(
        self,
        db: &'db dyn HirDatabase,
        acc: &mut Vec<AnyDiagnostic<'db>>,
        style_lints: bool,
    ) {
        match self {
            AssocItem::Function(func) => {
                GenericDef::Function(func).diagnostics(db, acc);
                DefWithBody::from(func).diagnostics(db, acc, style_lints);
            }
            AssocItem::Const(const_) => {
                GenericDef::Const(const_).diagnostics(db, acc);
                DefWithBody::from(const_).diagnostics(db, acc, style_lints);
            }
            AssocItem::TypeAlias(type_alias) => {
                GenericDef::TypeAlias(type_alias).diagnostics(db, acc);
                push_ty_diagnostics(
                    db,
                    acc,
                    db.type_for_type_alias_with_diagnostics(type_alias.id).1,
                    &db.type_alias_signature_with_source_map(type_alias.id).1,
                );
                for diag in hir_ty::diagnostics::incorrect_case(db, type_alias.id.into()) {
                    acc.push(diag.into());
                }
            }
        }
    }
}

impl HasVisibility for AssocItem {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility {
        match self {
            AssocItem::Function(f) => f.visibility(db),
            AssocItem::Const(c) => c.visibility(db),
            AssocItem::TypeAlias(t) => t.visibility(db),
        }
    }
}

impl From<AssocItem> for ModuleDef {
    fn from(assoc: AssocItem) -> Self {
        match assoc {
            AssocItem::Function(it) => ModuleDef::Function(it),
            AssocItem::Const(it) => ModuleDef::Const(it),
            AssocItem::TypeAlias(it) => ModuleDef::TypeAlias(it),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GenericDef {
    Function(Function),
    Adt(Adt),
    Trait(Trait),
    TraitAlias(TraitAlias),
    TypeAlias(TypeAlias),
    Impl(Impl),
    // consts can have type parameters from their parents (i.e. associated consts of traits)
    Const(Const),
    Static(Static),
}
impl_from!(
    Function,
    Adt(Struct, Enum, Union),
    Trait,
    TraitAlias,
    TypeAlias,
    Impl,
    Const,
    Static
    for GenericDef
);

impl GenericDef {
    pub fn params(self, db: &dyn HirDatabase) -> Vec<GenericParam> {
        let generics = db.generic_params(self.into());
        let ty_params = generics.iter_type_or_consts().map(|(local_id, _)| {
            let toc = TypeOrConstParam { id: TypeOrConstParamId { parent: self.into(), local_id } };
            match toc.split(db) {
                Either::Left(it) => GenericParam::ConstParam(it),
                Either::Right(it) => GenericParam::TypeParam(it),
            }
        });
        self.lifetime_params(db)
            .into_iter()
            .map(GenericParam::LifetimeParam)
            .chain(ty_params)
            .collect()
    }

    pub fn lifetime_params(self, db: &dyn HirDatabase) -> Vec<LifetimeParam> {
        let generics = db.generic_params(self.into());
        generics
            .iter_lt()
            .map(|(local_id, _)| LifetimeParam {
                id: LifetimeParamId { parent: self.into(), local_id },
            })
            .collect()
    }

    pub fn type_or_const_params(self, db: &dyn HirDatabase) -> Vec<TypeOrConstParam> {
        let generics = db.generic_params(self.into());
        generics
            .iter_type_or_consts()
            .map(|(local_id, _)| TypeOrConstParam {
                id: TypeOrConstParamId { parent: self.into(), local_id },
            })
            .collect()
    }

    fn id(self) -> GenericDefId {
        match self {
            GenericDef::Function(it) => it.id.into(),
            GenericDef::Adt(it) => it.into(),
            GenericDef::Trait(it) => it.id.into(),
            GenericDef::TraitAlias(it) => it.id.into(),
            GenericDef::TypeAlias(it) => it.id.into(),
            GenericDef::Impl(it) => it.id.into(),
            GenericDef::Const(it) => it.id.into(),
            GenericDef::Static(it) => it.id.into(),
        }
    }

    pub fn diagnostics<'db>(self, db: &'db dyn HirDatabase, acc: &mut Vec<AnyDiagnostic<'db>>) {
        let def = self.id();

        let generics = db.generic_params(def);

        if generics.is_empty() && generics.has_no_predicates() {
            return;
        }

        let source_map = match def {
            GenericDefId::AdtId(AdtId::EnumId(it)) => db.enum_signature_with_source_map(it).1,
            GenericDefId::AdtId(AdtId::StructId(it)) => db.struct_signature_with_source_map(it).1,
            GenericDefId::AdtId(AdtId::UnionId(it)) => db.union_signature_with_source_map(it).1,
            GenericDefId::ConstId(_) => return,
            GenericDefId::FunctionId(it) => db.function_signature_with_source_map(it).1,
            GenericDefId::ImplId(it) => db.impl_signature_with_source_map(it).1,
            GenericDefId::StaticId(_) => return,
            GenericDefId::TraitAliasId(it) => db.trait_alias_signature_with_source_map(it).1,
            GenericDefId::TraitId(it) => db.trait_signature_with_source_map(it).1,
            GenericDefId::TypeAliasId(it) => db.type_alias_signature_with_source_map(it).1,
        };

        expr_store_diagnostics(db, acc, &source_map);
        push_ty_diagnostics(db, acc, db.generic_defaults_with_diagnostics(def).1, &source_map);
        push_ty_diagnostics(
            db,
            acc,
            db.generic_predicates_without_parent_with_diagnostics(def).1,
            &source_map,
        );
        for (param_id, param) in generics.iter_type_or_consts() {
            if let TypeOrConstParamData::ConstParamData(_) = param {
                push_ty_diagnostics(
                    db,
                    acc,
                    db.const_param_ty_with_diagnostics(ConstParamId::from_unchecked(
                        TypeOrConstParamId { parent: def, local_id: param_id },
                    ))
                    .1,
                    &source_map,
                );
            }
        }
    }

    /// Returns a string describing the kind of this type.
    #[inline]
    pub fn description(self) -> &'static str {
        match self {
            GenericDef::Function(_) => "function",
            GenericDef::Adt(Adt::Struct(_)) => "struct",
            GenericDef::Adt(Adt::Enum(_)) => "enum",
            GenericDef::Adt(Adt::Union(_)) => "union",
            GenericDef::Trait(_) => "trait",
            GenericDef::TraitAlias(_) => "trait alias",
            GenericDef::TypeAlias(_) => "type alias",
            GenericDef::Impl(_) => "impl",
            GenericDef::Const(_) => "constant",
            GenericDef::Static(_) => "static",
        }
    }
}

// We cannot call this `Substitution` unfortunately...
#[derive(Debug)]
pub struct GenericSubstitution<'db> {
    def: GenericDefId,
    subst: Substitution,
    env: Arc<TraitEnvironment>,
    _pd: PhantomCovariantLifetime<'db>,
}

impl<'db> GenericSubstitution<'db> {
    fn new(def: GenericDefId, subst: Substitution, env: Arc<TraitEnvironment>) -> Self {
        Self { def, subst, env, _pd: PhantomCovariantLifetime::new() }
    }

    pub fn types(&self, db: &'db dyn HirDatabase) -> Vec<(Symbol, Type<'db>)> {
        let container = match self.def {
            GenericDefId::ConstId(id) => Some(id.lookup(db).container),
            GenericDefId::FunctionId(id) => Some(id.lookup(db).container),
            GenericDefId::TypeAliasId(id) => Some(id.lookup(db).container),
            _ => None,
        };
        let container_type_params = container
            .and_then(|container| match container {
                ItemContainerId::ImplId(container) => Some(container.into()),
                ItemContainerId::TraitId(container) => Some(container.into()),
                _ => None,
            })
            .map(|container| {
                db.generic_params(container)
                    .iter_type_or_consts()
                    .filter_map(|param| match param.1 {
                        TypeOrConstParamData::TypeParamData(param) => Some(param.name.clone()),
                        TypeOrConstParamData::ConstParamData(_) => None,
                    })
                    .collect::<Vec<_>>()
            });
        let generics = db.generic_params(self.def);
        let type_params = generics.iter_type_or_consts().filter_map(|param| match param.1 {
            TypeOrConstParamData::TypeParamData(param) => Some(param.name.clone()),
            TypeOrConstParamData::ConstParamData(_) => None,
        });
        let parent_len = self.subst.len(Interner)
            - generics
                .iter_type_or_consts()
                .filter(|g| matches!(g.1, TypeOrConstParamData::TypeParamData(..)))
                .count();
        let container_params = self.subst.as_slice(Interner)[..parent_len]
            .iter()
            .filter_map(|param| param.ty(Interner).cloned())
            .zip(container_type_params.into_iter().flatten());
        let self_params = self.subst.as_slice(Interner)[parent_len..]
            .iter()
            .filter_map(|param| param.ty(Interner).cloned())
            .zip(type_params);
        container_params
            .chain(self_params)
            .filter_map(|(ty, name)| {
                Some((
                    name?.symbol().clone(),
                    Type { ty, env: self.env.clone(), _pd: PhantomCovariantLifetime::new() },
                ))
            })
            .collect()
    }
}

/// A single local definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Local {
    pub(crate) parent: DefWithBodyId,
    pub(crate) binding_id: BindingId,
}

pub struct LocalSource {
    pub local: Local,
    pub source: InFile<Either<ast::IdentPat, ast::SelfParam>>,
}

impl LocalSource {
    pub fn as_ident_pat(&self) -> Option<&ast::IdentPat> {
        match &self.source.value {
            Either::Left(it) => Some(it),
            Either::Right(_) => None,
        }
    }

    pub fn into_ident_pat(self) -> Option<ast::IdentPat> {
        match self.source.value {
            Either::Left(it) => Some(it),
            Either::Right(_) => None,
        }
    }

    pub fn original_file(&self, db: &dyn HirDatabase) -> EditionedFileId {
        self.source.file_id.original_file(db)
    }

    pub fn file(&self) -> HirFileId {
        self.source.file_id
    }

    pub fn name(&self) -> Option<InFile<ast::Name>> {
        self.source.as_ref().map(|it| it.name()).transpose()
    }

    pub fn syntax(&self) -> &SyntaxNode {
        self.source.value.syntax()
    }

    pub fn syntax_ptr(self) -> InFile<SyntaxNodePtr> {
        self.source.map(|it| SyntaxNodePtr::new(it.syntax()))
    }
}

impl Local {
    pub fn is_param(self, db: &dyn HirDatabase) -> bool {
        // FIXME: This parses!
        let src = self.primary_source(db);
        match src.source.value {
            Either::Left(pat) => pat
                .syntax()
                .ancestors()
                .map(|it| it.kind())
                .take_while(|&kind| ast::Pat::can_cast(kind) || ast::Param::can_cast(kind))
                .any(ast::Param::can_cast),
            Either::Right(_) => true,
        }
    }

    pub fn as_self_param(self, db: &dyn HirDatabase) -> Option<SelfParam> {
        match self.parent {
            DefWithBodyId::FunctionId(func) if self.is_self(db) => Some(SelfParam { func }),
            _ => None,
        }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        let body = db.body(self.parent);
        body[self.binding_id].name.clone()
    }

    pub fn is_self(self, db: &dyn HirDatabase) -> bool {
        self.name(db) == sym::self_
    }

    pub fn is_mut(self, db: &dyn HirDatabase) -> bool {
        let body = db.body(self.parent);
        body[self.binding_id].mode == BindingAnnotation::Mutable
    }

    pub fn is_ref(self, db: &dyn HirDatabase) -> bool {
        let body = db.body(self.parent);
        matches!(body[self.binding_id].mode, BindingAnnotation::Ref | BindingAnnotation::RefMut)
    }

    pub fn parent(self, _db: &dyn HirDatabase) -> DefWithBody {
        self.parent.into()
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        self.parent(db).module(db)
    }

    pub fn ty(self, db: &dyn HirDatabase) -> Type<'_> {
        let def = self.parent;
        let infer = db.infer(def);
        let ty = infer[self.binding_id].clone();
        Type::new(db, def, ty)
    }

    /// All definitions for this local. Example: `let (a$0, _) | (_, a$0) = it;`
    pub fn sources(self, db: &dyn HirDatabase) -> Vec<LocalSource> {
        let (body, source_map) = db.body_with_source_map(self.parent);
        match body.self_param.zip(source_map.self_param_syntax()) {
            Some((param, source)) if param == self.binding_id => {
                let root = source.file_syntax(db);
                vec![LocalSource {
                    local: self,
                    source: source.map(|ast| Either::Right(ast.to_node(&root))),
                }]
            }
            _ => source_map
                .patterns_for_binding(self.binding_id)
                .iter()
                .map(|&definition| {
                    let src = source_map.pat_syntax(definition).unwrap(); // Hmm...
                    let root = src.file_syntax(db);
                    LocalSource {
                        local: self,
                        source: src.map(|ast| match ast.to_node(&root) {
                            Either::Right(ast::Pat::IdentPat(it)) => Either::Left(it),
                            _ => unreachable!("local with non ident-pattern"),
                        }),
                    }
                })
                .collect(),
        }
    }

    /// The leftmost definition for this local. Example: `let (a$0, _) | (_, a) = it;`
    pub fn primary_source(self, db: &dyn HirDatabase) -> LocalSource {
        let (body, source_map) = db.body_with_source_map(self.parent);
        match body.self_param.zip(source_map.self_param_syntax()) {
            Some((param, source)) if param == self.binding_id => {
                let root = source.file_syntax(db);
                LocalSource {
                    local: self,
                    source: source.map(|ast| Either::Right(ast.to_node(&root))),
                }
            }
            _ => source_map
                .patterns_for_binding(self.binding_id)
                .first()
                .map(|&definition| {
                    let src = source_map.pat_syntax(definition).unwrap(); // Hmm...
                    let root = src.file_syntax(db);
                    LocalSource {
                        local: self,
                        source: src.map(|ast| match ast.to_node(&root) {
                            Either::Right(ast::Pat::IdentPat(it)) => Either::Left(it),
                            _ => unreachable!("local with non ident-pattern"),
                        }),
                    }
                })
                .unwrap(),
        }
    }
}

impl PartialOrd for Local {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Local {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.binding_id.cmp(&other.binding_id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeriveHelper {
    pub(crate) derive: MacroId,
    pub(crate) idx: u32,
}

impl DeriveHelper {
    pub fn derive(&self) -> Macro {
        Macro { id: self.derive }
    }

    pub fn name(&self, db: &dyn HirDatabase) -> Name {
        match self.derive {
            makro @ MacroId::Macro2Id(_) => db
                .attrs(makro.into())
                .parse_rustc_builtin_macro()
                .and_then(|(_, helpers)| helpers.get(self.idx as usize).cloned()),
            MacroId::MacroRulesId(_) => None,
            makro @ MacroId::ProcMacroId(_) => db
                .attrs(makro.into())
                .parse_proc_macro_derive()
                .and_then(|(_, helpers)| helpers.get(self.idx as usize).cloned()),
        }
        .unwrap_or_else(Name::missing)
    }
}

// FIXME: Wrong name? This is could also be a registered attribute
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BuiltinAttr {
    krate: Option<base_db::Crate>,
    idx: u32,
}

impl BuiltinAttr {
    // FIXME: consider crates\hir_def\src\nameres\attr_resolution.rs?
    pub(crate) fn by_name(db: &dyn HirDatabase, krate: Crate, name: &str) -> Option<Self> {
        if let builtin @ Some(_) = Self::builtin(name) {
            return builtin;
        }
        let idx = crate_def_map(db, krate.id)
            .registered_attrs()
            .iter()
            .position(|it| it.as_str() == name)? as u32;
        Some(BuiltinAttr { krate: Some(krate.id), idx })
    }

    fn builtin(name: &str) -> Option<Self> {
        hir_expand::inert_attr_macro::find_builtin_attr_idx(&Symbol::intern(name))
            .map(|idx| BuiltinAttr { krate: None, idx: idx as u32 })
    }

    pub fn name(&self, db: &dyn HirDatabase) -> Name {
        match self.krate {
            Some(krate) => Name::new_symbol_root(
                crate_def_map(db, krate).registered_attrs()[self.idx as usize].clone(),
            ),
            None => Name::new_symbol_root(Symbol::intern(
                hir_expand::inert_attr_macro::INERT_ATTRIBUTES[self.idx as usize].name,
            )),
        }
    }

    pub fn template(&self, _: &dyn HirDatabase) -> Option<AttributeTemplate> {
        match self.krate {
            Some(_) => None,
            None => {
                Some(hir_expand::inert_attr_macro::INERT_ATTRIBUTES[self.idx as usize].template)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ToolModule {
    krate: base_db::Crate,
    idx: u32,
}

impl ToolModule {
    pub(crate) fn by_name(db: &dyn HirDatabase, krate: Crate, name: &str) -> Option<Self> {
        let krate = krate.id;
        let idx =
            crate_def_map(db, krate).registered_tools().iter().position(|it| it.as_str() == name)?
                as u32;
        Some(ToolModule { krate, idx })
    }

    pub fn name(&self, db: &dyn HirDatabase) -> Name {
        Name::new_symbol_root(
            crate_def_map(db, self.krate).registered_tools()[self.idx as usize].clone(),
        )
    }

    pub fn krate(&self) -> Crate {
        Crate { id: self.krate }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Label {
    pub(crate) parent: DefWithBodyId,
    pub(crate) label_id: LabelId,
}

impl Label {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        self.parent(db).module(db)
    }

    pub fn parent(self, _db: &dyn HirDatabase) -> DefWithBody {
        self.parent.into()
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        let body = db.body(self.parent);
        body[self.label_id].name.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GenericParam {
    TypeParam(TypeParam),
    ConstParam(ConstParam),
    LifetimeParam(LifetimeParam),
}
impl_from!(TypeParam, ConstParam, LifetimeParam for GenericParam);

impl GenericParam {
    pub fn module(self, db: &dyn HirDatabase) -> Module {
        match self {
            GenericParam::TypeParam(it) => it.module(db),
            GenericParam::ConstParam(it) => it.module(db),
            GenericParam::LifetimeParam(it) => it.module(db),
        }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        match self {
            GenericParam::TypeParam(it) => it.name(db),
            GenericParam::ConstParam(it) => it.name(db),
            GenericParam::LifetimeParam(it) => it.name(db),
        }
    }

    pub fn parent(self) -> GenericDef {
        match self {
            GenericParam::TypeParam(it) => it.id.parent().into(),
            GenericParam::ConstParam(it) => it.id.parent().into(),
            GenericParam::LifetimeParam(it) => it.id.parent.into(),
        }
    }

    pub fn variance(self, db: &dyn HirDatabase) -> Option<Variance> {
        let parent = match self {
            GenericParam::TypeParam(it) => it.id.parent(),
            // const parameters are always invariant
            GenericParam::ConstParam(_) => return None,
            GenericParam::LifetimeParam(it) => it.id.parent,
        };
        let generics = hir_ty::generics::generics(db, parent);
        let index = match self {
            GenericParam::TypeParam(it) => generics.type_or_const_param_idx(it.id.into())?,
            GenericParam::ConstParam(_) => return None,
            GenericParam::LifetimeParam(it) => generics.lifetime_idx(it.id)?,
        };
        db.variances_of(parent)?.get(index).copied()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeParam {
    pub(crate) id: TypeParamId,
}

impl TypeParam {
    pub fn merge(self) -> TypeOrConstParam {
        TypeOrConstParam { id: self.id.into() }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        self.merge().name(db)
    }

    pub fn parent(self, _db: &dyn HirDatabase) -> GenericDef {
        self.id.parent().into()
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        self.id.parent().module(db).into()
    }

    /// Is this type parameter implicitly introduced (eg. `Self` in a trait or an `impl Trait`
    /// argument)?
    pub fn is_implicit(self, db: &dyn HirDatabase) -> bool {
        let params = db.generic_params(self.id.parent());
        let data = &params[self.id.local_id()];
        match data.type_param().unwrap().provenance {
            TypeParamProvenance::TypeParamList => false,
            TypeParamProvenance::TraitSelf | TypeParamProvenance::ArgumentImplTrait => true,
        }
    }

    pub fn ty(self, db: &dyn HirDatabase) -> Type<'_> {
        let resolver = self.id.parent().resolver(db);
        let ty =
            TyKind::Placeholder(hir_ty::to_placeholder_idx(db, self.id.into())).intern(Interner);
        Type::new_with_resolver_inner(db, &resolver, ty)
    }

    /// FIXME: this only lists trait bounds from the item defining the type
    /// parameter, not additional bounds that might be added e.g. by a method if
    /// the parameter comes from an impl!
    pub fn trait_bounds(self, db: &dyn HirDatabase) -> Vec<Trait> {
        db.generic_predicates_for_param(self.id.parent(), self.id.into(), None)
            .iter()
            .filter_map(|pred| match &pred.skip_binders().skip_binders() {
                hir_ty::WhereClause::Implemented(trait_ref) => {
                    Some(Trait::from(trait_ref.hir_trait_id()))
                }
                _ => None,
            })
            .collect()
    }

    pub fn default(self, db: &dyn HirDatabase) -> Option<Type<'_>> {
        let ty = generic_arg_from_param(db, self.id.into())?;
        let resolver = self.id.parent().resolver(db);
        match ty.data(Interner) {
            GenericArgData::Ty(it) if *it.kind(Interner) != TyKind::Error => {
                Some(Type::new_with_resolver_inner(db, &resolver, it.clone()))
            }
            _ => None,
        }
    }

    pub fn is_unstable(self, db: &dyn HirDatabase) -> bool {
        db.attrs(GenericParamId::from(self.id).into()).is_unstable()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LifetimeParam {
    pub(crate) id: LifetimeParamId,
}

impl LifetimeParam {
    pub fn name(self, db: &dyn HirDatabase) -> Name {
        let params = db.generic_params(self.id.parent);
        params[self.id.local_id].name.clone()
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        self.id.parent.module(db).into()
    }

    pub fn parent(self, _db: &dyn HirDatabase) -> GenericDef {
        self.id.parent.into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConstParam {
    pub(crate) id: ConstParamId,
}

impl ConstParam {
    pub fn merge(self) -> TypeOrConstParam {
        TypeOrConstParam { id: self.id.into() }
    }

    pub fn name(self, db: &dyn HirDatabase) -> Name {
        let params = db.generic_params(self.id.parent());
        match params[self.id.local_id()].name() {
            Some(it) => it.clone(),
            None => {
                never!();
                Name::missing()
            }
        }
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        self.id.parent().module(db).into()
    }

    pub fn parent(self, _db: &dyn HirDatabase) -> GenericDef {
        self.id.parent().into()
    }

    pub fn ty(self, db: &dyn HirDatabase) -> Type<'_> {
        Type::new(db, self.id.parent(), db.const_param_ty(self.id))
    }

    pub fn default(
        self,
        db: &dyn HirDatabase,
        display_target: DisplayTarget,
    ) -> Option<ast::ConstArg> {
        let arg = generic_arg_from_param(db, self.id.into())?;
        known_const_to_ast(arg.constant(Interner)?, db, display_target)
    }
}

fn generic_arg_from_param(db: &dyn HirDatabase, id: TypeOrConstParamId) -> Option<GenericArg> {
    let local_idx = hir_ty::param_idx(db, id)?;
    let defaults = db.generic_defaults(id.parent);
    let ty = defaults.get(local_idx)?.clone();
    let full_subst = TyBuilder::placeholder_subst(db, id.parent);
    let subst = &full_subst.as_slice(Interner)[..local_idx];
    Some(ty.substitute(Interner, &subst))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeOrConstParam {
    pub(crate) id: TypeOrConstParamId,
}

impl TypeOrConstParam {
    pub fn name(self, db: &dyn HirDatabase) -> Name {
        let params = db.generic_params(self.id.parent);
        match params[self.id.local_id].name() {
            Some(n) => n.clone(),
            _ => Name::missing(),
        }
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        self.id.parent.module(db).into()
    }

    pub fn parent(self, _db: &dyn HirDatabase) -> GenericDef {
        self.id.parent.into()
    }

    pub fn split(self, db: &dyn HirDatabase) -> Either<ConstParam, TypeParam> {
        let params = db.generic_params(self.id.parent);
        match &params[self.id.local_id] {
            TypeOrConstParamData::TypeParamData(_) => {
                Either::Right(TypeParam { id: TypeParamId::from_unchecked(self.id) })
            }
            TypeOrConstParamData::ConstParamData(_) => {
                Either::Left(ConstParam { id: ConstParamId::from_unchecked(self.id) })
            }
        }
    }

    pub fn ty(self, db: &dyn HirDatabase) -> Type<'_> {
        match self.split(db) {
            Either::Left(it) => it.ty(db),
            Either::Right(it) => it.ty(db),
        }
    }

    pub fn as_type_param(self, db: &dyn HirDatabase) -> Option<TypeParam> {
        let params = db.generic_params(self.id.parent);
        match &params[self.id.local_id] {
            TypeOrConstParamData::TypeParamData(_) => {
                Some(TypeParam { id: TypeParamId::from_unchecked(self.id) })
            }
            TypeOrConstParamData::ConstParamData(_) => None,
        }
    }

    pub fn as_const_param(self, db: &dyn HirDatabase) -> Option<ConstParam> {
        let params = db.generic_params(self.id.parent);
        match &params[self.id.local_id] {
            TypeOrConstParamData::TypeParamData(_) => None,
            TypeOrConstParamData::ConstParamData(_) => {
                Some(ConstParam { id: ConstParamId::from_unchecked(self.id) })
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Impl {
    pub(crate) id: ImplId,
}

impl Impl {
    pub fn all_in_crate(db: &dyn HirDatabase, krate: Crate) -> Vec<Impl> {
        let inherent = db.inherent_impls_in_crate(krate.id);
        let trait_ = db.trait_impls_in_crate(krate.id);

        inherent.all_impls().chain(trait_.all_impls()).map(Self::from).collect()
    }

    pub fn all_in_module(db: &dyn HirDatabase, module: Module) -> Vec<Impl> {
        module.id.def_map(db)[module.id.local_id].scope.impls().map(Into::into).collect()
    }

    pub fn all_for_type<'db>(
        db: &'db dyn HirDatabase,
        Type { ty, env, _pd: _ }: Type<'db>,
    ) -> Vec<Impl> {
        let def_crates = match method_resolution::def_crates(db, &ty, env.krate) {
            Some(def_crates) => def_crates,
            None => return Vec::new(),
        };

        let filter = |impl_def: &Impl| {
            let self_ty = impl_def.self_ty(db);
            let rref = self_ty.remove_ref();
            ty.equals_ctor(rref.as_ref().map_or(&self_ty.ty, |it| &it.ty))
        };

        let fp = TyFingerprint::for_inherent_impl(&ty);
        let fp = match fp {
            Some(fp) => fp,
            None => return Vec::new(),
        };

        let mut all = Vec::new();
        def_crates.iter().for_each(|&id| {
            all.extend(
                db.inherent_impls_in_crate(id)
                    .for_self_ty(&ty)
                    .iter()
                    .cloned()
                    .map(Self::from)
                    .filter(filter),
            )
        });

        for id in def_crates
            .iter()
            .flat_map(|&id| Crate { id }.transitive_reverse_dependencies(db))
            .map(|Crate { id }| id)
        {
            all.extend(
                db.trait_impls_in_crate(id)
                    .for_self_ty_without_blanket_impls(fp)
                    .map(Self::from)
                    .filter(filter),
            );
        }

        if let Some(block) = ty.adt_id(Interner).and_then(|def| def.0.module(db).containing_block())
        {
            if let Some(inherent_impls) = db.inherent_impls_in_block(block) {
                all.extend(
                    inherent_impls.for_self_ty(&ty).iter().cloned().map(Self::from).filter(filter),
                );
            }
            if let Some(trait_impls) = db.trait_impls_in_block(block) {
                all.extend(
                    trait_impls
                        .for_self_ty_without_blanket_impls(fp)
                        .map(Self::from)
                        .filter(filter),
                );
            }
        }

        all
    }

    pub fn all_for_trait(db: &dyn HirDatabase, trait_: Trait) -> Vec<Impl> {
        let module = trait_.module(db);
        let krate = module.krate();
        let mut all = Vec::new();
        for Crate { id } in krate.transitive_reverse_dependencies(db) {
            let impls = db.trait_impls_in_crate(id);
            all.extend(impls.for_trait(trait_.id).map(Self::from))
        }
        if let Some(block) = module.id.containing_block() {
            if let Some(trait_impls) = db.trait_impls_in_block(block) {
                all.extend(trait_impls.for_trait(trait_.id).map(Self::from));
            }
        }
        all
    }

    pub fn trait_(self, db: &dyn HirDatabase) -> Option<Trait> {
        let trait_ref = db.impl_trait(self.id)?;
        let id = trait_ref.skip_binders().hir_trait_id();
        Some(Trait { id })
    }

    pub fn trait_ref(self, db: &dyn HirDatabase) -> Option<TraitRef<'_>> {
        let substs = TyBuilder::placeholder_subst(db, self.id);
        let trait_ref = db.impl_trait(self.id)?.substitute(Interner, &substs);
        let resolver = self.id.resolver(db);
        Some(TraitRef::new_with_resolver(db, &resolver, trait_ref))
    }

    pub fn self_ty(self, db: &dyn HirDatabase) -> Type<'_> {
        let resolver = self.id.resolver(db);
        let substs = TyBuilder::placeholder_subst(db, self.id);
        let ty = db.impl_self_ty(self.id).substitute(Interner, &substs);
        Type::new_with_resolver_inner(db, &resolver, ty)
    }

    pub fn items(self, db: &dyn HirDatabase) -> Vec<AssocItem> {
        self.id.impl_items(db).items.iter().map(|&(_, it)| it.into()).collect()
    }

    pub fn is_negative(self, db: &dyn HirDatabase) -> bool {
        db.impl_signature(self.id).flags.contains(ImplFlags::NEGATIVE)
    }

    pub fn is_unsafe(self, db: &dyn HirDatabase) -> bool {
        db.impl_signature(self.id).flags.contains(ImplFlags::UNSAFE)
    }

    pub fn module(self, db: &dyn HirDatabase) -> Module {
        self.id.lookup(db).container.into()
    }

    pub fn as_builtin_derive_path(self, db: &dyn HirDatabase) -> Option<InMacroFile<ast::Path>> {
        let src = self.source(db)?;

        let macro_file = src.file_id.macro_file()?;
        let loc = macro_file.lookup(db);
        let (derive_attr, derive_index) = match loc.kind {
            MacroCallKind::Derive { ast_id, derive_attr_index, derive_index, .. } => {
                let module_id = self.id.lookup(db).container;
                (
                    crate_def_map(db, module_id.krate())[module_id.local_id]
                        .scope
                        .derive_macro_invoc(ast_id, derive_attr_index)?,
                    derive_index,
                )
            }
            _ => return None,
        };
        let path = db
            .parse_macro_expansion(derive_attr)
            .value
            .0
            .syntax_node()
            .children()
            .nth(derive_index as usize)
            .and_then(<ast::Attr as AstNode>::cast)
            .and_then(|it| it.path())?;
        Some(InMacroFile { file_id: derive_attr, value: path })
    }

    pub fn check_orphan_rules(self, db: &dyn HirDatabase) -> bool {
        check_orphan_rules(db, self.id)
    }

    fn all_macro_calls(&self, db: &dyn HirDatabase) -> Box<[(AstId<ast::Item>, MacroCallId)]> {
        self.id.impl_items(db).macro_calls.to_vec().into_boxed_slice()
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct TraitRef<'db> {
    env: Arc<TraitEnvironment>,
    trait_ref: hir_ty::TraitRef,
    _pd: PhantomCovariantLifetime<'db>,
}

impl<'db> TraitRef<'db> {
    pub(crate) fn new_with_resolver(
        db: &'db dyn HirDatabase,
        resolver: &Resolver<'_>,
        trait_ref: hir_ty::TraitRef,
    ) -> Self {
        let env = resolver
            .generic_def()
            .map_or_else(|| TraitEnvironment::empty(resolver.krate()), |d| db.trait_environment(d));
        TraitRef { env, trait_ref, _pd: PhantomCovariantLifetime::new() }
    }

    pub fn trait_(&self) -> Trait {
        let id = self.trait_ref.hir_trait_id();
        Trait { id }
    }

    pub fn self_ty(&self) -> Type<'_> {
        let ty = self.trait_ref.self_type_parameter(Interner);
        Type { env: self.env.clone(), ty, _pd: PhantomCovariantLifetime::new() }
    }

    /// Returns `idx`-th argument of this trait reference if it is a type argument. Note that the
    /// first argument is the `Self` type.
    pub fn get_type_argument(&self, idx: usize) -> Option<Type<'db>> {
        self.trait_ref
            .substitution
            .as_slice(Interner)
            .get(idx)
            .and_then(|arg| arg.ty(Interner))
            .cloned()
            .map(|ty| Type { env: self.env.clone(), ty, _pd: PhantomCovariantLifetime::new() })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Closure {
    id: ClosureId,
    subst: Substitution,
}

impl From<Closure> for ClosureId {
    fn from(value: Closure) -> Self {
        value.id
    }
}

impl Closure {
    fn as_ty(self) -> Ty {
        TyKind::Closure(self.id, self.subst).intern(Interner)
    }

    pub fn display_with_id(&self, db: &dyn HirDatabase, display_target: DisplayTarget) -> String {
        self.clone()
            .as_ty()
            .display(db, display_target)
            .with_closure_style(ClosureStyle::ClosureWithId)
            .to_string()
    }

    pub fn display_with_impl(&self, db: &dyn HirDatabase, display_target: DisplayTarget) -> String {
        self.clone()
            .as_ty()
            .display(db, display_target)
            .with_closure_style(ClosureStyle::ImplFn)
            .to_string()
    }

    pub fn captured_items(&self, db: &dyn HirDatabase) -> Vec<ClosureCapture> {
        let owner = db.lookup_intern_closure((self.id).into()).0;
        let infer = &db.infer(owner);
        let info = infer.closure_info(&self.id);
        info.0
            .iter()
            .cloned()
            .map(|capture| ClosureCapture { owner, closure: self.id, capture })
            .collect()
    }

    pub fn capture_types<'db>(&self, db: &'db dyn HirDatabase) -> Vec<Type<'db>> {
        let owner = db.lookup_intern_closure((self.id).into()).0;
        let infer = &db.infer(owner);
        let (captures, _) = infer.closure_info(&self.id);
        captures
            .iter()
            .map(|capture| Type {
                env: db.trait_environment_for_body(owner),
                ty: capture.ty(&self.subst),
                _pd: PhantomCovariantLifetime::new(),
            })
            .collect()
    }

    pub fn fn_trait(&self, db: &dyn HirDatabase) -> FnTrait {
        let owner = db.lookup_intern_closure((self.id).into()).0;
        let infer = &db.infer(owner);
        let info = infer.closure_info(&self.id);
        info.1
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureCapture {
    owner: DefWithBodyId,
    closure: ClosureId,
    capture: hir_ty::CapturedItem,
}

impl ClosureCapture {
    pub fn local(&self) -> Local {
        Local { parent: self.owner, binding_id: self.capture.local() }
    }

    /// Returns whether this place has any field (aka. non-deref) projections.
    pub fn has_field_projections(&self) -> bool {
        self.capture.has_field_projections()
    }

    pub fn usages(&self) -> CaptureUsages {
        CaptureUsages { parent: self.owner, spans: self.capture.spans() }
    }

    pub fn kind(&self) -> CaptureKind {
        match self.capture.kind() {
            hir_ty::CaptureKind::ByRef(
                hir_ty::mir::BorrowKind::Shallow | hir_ty::mir::BorrowKind::Shared,
            ) => CaptureKind::SharedRef,
            hir_ty::CaptureKind::ByRef(hir_ty::mir::BorrowKind::Mut {
                kind: MutBorrowKind::ClosureCapture,
            }) => CaptureKind::UniqueSharedRef,
            hir_ty::CaptureKind::ByRef(hir_ty::mir::BorrowKind::Mut {
                kind: MutBorrowKind::Default | MutBorrowKind::TwoPhasedBorrow,
            }) => CaptureKind::MutableRef,
            hir_ty::CaptureKind::ByValue => CaptureKind::Move,
        }
    }

    /// Converts the place to a name that can be inserted into source code.
    pub fn place_to_name(&self, db: &dyn HirDatabase) -> String {
        self.capture.place_to_name(self.owner, db)
    }

    pub fn display_place_source_code(&self, db: &dyn HirDatabase) -> String {
        self.capture.display_place_source_code(self.owner, db)
    }

    pub fn display_place(&self, db: &dyn HirDatabase) -> String {
        self.capture.display_place(self.owner, db)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CaptureKind {
    SharedRef,
    UniqueSharedRef,
    MutableRef,
    Move,
}

#[derive(Debug, Clone)]
pub struct CaptureUsages {
    parent: DefWithBodyId,
    spans: SmallVec<[mir::MirSpan; 3]>,
}

impl CaptureUsages {
    pub fn sources(&self, db: &dyn HirDatabase) -> Vec<CaptureUsageSource> {
        let (body, source_map) = db.body_with_source_map(self.parent);

        let mut result = Vec::with_capacity(self.spans.len());
        for &span in self.spans.iter() {
            let is_ref = span.is_ref_span(&body);
            match span {
                mir::MirSpan::ExprId(expr) => {
                    if let Ok(expr) = source_map.expr_syntax(expr) {
                        result.push(CaptureUsageSource { is_ref, source: expr })
                    }
                }
                mir::MirSpan::PatId(pat) => {
                    if let Ok(pat) = source_map.pat_syntax(pat) {
                        result.push(CaptureUsageSource { is_ref, source: pat });
                    }
                }
                mir::MirSpan::BindingId(binding) => result.extend(
                    source_map
                        .patterns_for_binding(binding)
                        .iter()
                        .filter_map(|&pat| source_map.pat_syntax(pat).ok())
                        .map(|pat| CaptureUsageSource { is_ref, source: pat }),
                ),
                mir::MirSpan::SelfParam | mir::MirSpan::Unknown => {
                    unreachable!("invalid capture usage span")
                }
            }
        }
        result
    }
}

#[derive(Debug)]
pub struct CaptureUsageSource {
    is_ref: bool,
    source: InFile<AstPtr<Either<ast::Expr, ast::Pat>>>,
}

impl CaptureUsageSource {
    pub fn source(&self) -> AstPtr<Either<ast::Expr, ast::Pat>> {
        self.source.value
    }

    pub fn file_id(&self) -> HirFileId {
        self.source.file_id
    }

    pub fn is_ref(&self) -> bool {
        self.is_ref
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Type<'db> {
    env: Arc<TraitEnvironment>,
    ty: Ty,
    _pd: PhantomCovariantLifetime<'db>,
}

impl<'db> Type<'db> {
    pub(crate) fn new_with_resolver(
        db: &'db dyn HirDatabase,
        resolver: &Resolver<'_>,
        ty: Ty,
    ) -> Self {
        Type::new_with_resolver_inner(db, resolver, ty)
    }

    pub(crate) fn new_with_resolver_inner(
        db: &'db dyn HirDatabase,
        resolver: &Resolver<'_>,
        ty: Ty,
    ) -> Self {
        let environment = resolver
            .generic_def()
            .map_or_else(|| TraitEnvironment::empty(resolver.krate()), |d| db.trait_environment(d));
        Type { env: environment, ty, _pd: PhantomCovariantLifetime::new() }
    }

    pub(crate) fn new_for_crate(krate: base_db::Crate, ty: Ty) -> Self {
        Type { env: TraitEnvironment::empty(krate), ty, _pd: PhantomCovariantLifetime::new() }
    }

    fn new(db: &'db dyn HirDatabase, lexical_env: impl HasResolver, ty: Ty) -> Self {
        let resolver = lexical_env.resolver(db);
        let environment = resolver
            .generic_def()
            .map_or_else(|| TraitEnvironment::empty(resolver.krate()), |d| db.trait_environment(d));
        Type { env: environment, ty, _pd: PhantomCovariantLifetime::new() }
    }

    fn from_def(db: &'db dyn HirDatabase, def: impl Into<TyDefId> + HasResolver) -> Self {
        let ty = db.ty(def.into());
        let substs = TyBuilder::unknown_subst(
            db,
            match def.into() {
                TyDefId::AdtId(it) => GenericDefId::AdtId(it),
                TyDefId::TypeAliasId(it) => GenericDefId::TypeAliasId(it),
                TyDefId::BuiltinType(_) => return Type::new(db, def, ty.skip_binders().clone()),
            },
        );
        Type::new(db, def, ty.substitute(Interner, &substs))
    }

    fn from_def_placeholders(
        db: &'db dyn HirDatabase,
        def: impl Into<TyDefId> + HasResolver,
    ) -> Self {
        let ty = db.ty(def.into());
        let substs = TyBuilder::placeholder_subst(
            db,
            match def.into() {
                TyDefId::AdtId(it) => GenericDefId::AdtId(it),
                TyDefId::TypeAliasId(it) => GenericDefId::TypeAliasId(it),
                TyDefId::BuiltinType(_) => return Type::new(db, def, ty.skip_binders().clone()),
            },
        );
        Type::new(db, def, ty.substitute(Interner, &substs))
    }

    fn from_value_def(
        db: &'db dyn HirDatabase,
        def: impl Into<ValueTyDefId> + HasResolver,
    ) -> Self {
        let Some(ty) = db.value_ty(def.into()) else {
            return Type::new(db, def, TyKind::Error.intern(Interner));
        };
        let substs = TyBuilder::unknown_subst(
            db,
            match def.into() {
                ValueTyDefId::ConstId(it) => GenericDefId::ConstId(it),
                ValueTyDefId::FunctionId(it) => GenericDefId::FunctionId(it),
                ValueTyDefId::StructId(it) => GenericDefId::AdtId(AdtId::StructId(it)),
                ValueTyDefId::UnionId(it) => GenericDefId::AdtId(AdtId::UnionId(it)),
                ValueTyDefId::EnumVariantId(it) => {
                    GenericDefId::AdtId(AdtId::EnumId(it.lookup(db).parent))
                }
                ValueTyDefId::StaticId(_) => return Type::new(db, def, ty.skip_binders().clone()),
            },
        );
        Type::new(db, def, ty.substitute(Interner, &substs))
    }

    pub fn new_slice(ty: Self) -> Self {
        Type { env: ty.env, ty: TyBuilder::slice(ty.ty), _pd: PhantomCovariantLifetime::new() }
    }

    pub fn new_tuple(krate: base_db::Crate, tys: &[Self]) -> Self {
        let tys = tys.iter().map(|it| it.ty.clone());
        Type {
            env: TraitEnvironment::empty(krate),
            ty: TyBuilder::tuple_with(tys),
            _pd: PhantomCovariantLifetime::new(),
        }
    }

    pub fn is_unit(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Tuple(0, ..))
    }

    pub fn is_bool(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Scalar(Scalar::Bool))
    }

    pub fn is_str(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Str)
    }

    pub fn is_never(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Never)
    }

    pub fn is_mutable_reference(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Ref(hir_ty::Mutability::Mut, ..))
    }

    pub fn is_reference(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Ref(..))
    }

    pub fn contains_reference(&self, db: &'db dyn HirDatabase) -> bool {
        return go(db, self.env.krate, &self.ty);

        fn go(db: &dyn HirDatabase, krate: base_db::Crate, ty: &Ty) -> bool {
            match ty.kind(Interner) {
                // Reference itself
                TyKind::Ref(_, _, _) => true,

                // For non-phantom_data adts we check variants/fields as well as generic parameters
                TyKind::Adt(adt_id, substitution)
                    if !db.adt_datum(krate, *adt_id).flags.phantom_data =>
                {
                    let adt_datum = &db.adt_datum(krate, *adt_id);
                    let adt_datum_bound =
                        adt_datum.binders.clone().substitute(Interner, substitution);
                    adt_datum_bound
                        .variants
                        .into_iter()
                        .flat_map(|variant| variant.fields.into_iter())
                        .any(|ty| go(db, krate, &ty))
                        || substitution
                            .iter(Interner)
                            .filter_map(|x| x.ty(Interner))
                            .any(|ty| go(db, krate, ty))
                }
                // And for `PhantomData<T>`, we check `T`.
                TyKind::Adt(_, substitution)
                | TyKind::Tuple(_, substitution)
                | TyKind::OpaqueType(_, substitution)
                | TyKind::AssociatedType(_, substitution)
                | TyKind::FnDef(_, substitution) => substitution
                    .iter(Interner)
                    .filter_map(|x| x.ty(Interner))
                    .any(|ty| go(db, krate, ty)),

                // For `[T]` or `*T` we check `T`
                TyKind::Array(ty, _) | TyKind::Slice(ty) | TyKind::Raw(_, ty) => go(db, krate, ty),

                // Consider everything else as not reference
                _ => false,
            }
        }
    }

    pub fn as_reference(&self) -> Option<(Type<'db>, Mutability)> {
        let (ty, _lt, m) = self.ty.as_reference()?;
        let m = Mutability::from_mutable(matches!(m, hir_ty::Mutability::Mut));
        Some((self.derived(ty.clone()), m))
    }

    pub fn add_reference(&self, mutability: Mutability) -> Self {
        let ty_mutability = match mutability {
            Mutability::Shared => hir_ty::Mutability::Not,
            Mutability::Mut => hir_ty::Mutability::Mut,
        };
        self.derived(TyKind::Ref(ty_mutability, error_lifetime(), self.ty.clone()).intern(Interner))
    }

    pub fn is_slice(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Slice(..))
    }

    pub fn is_usize(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Scalar(Scalar::Uint(UintTy::Usize)))
    }

    pub fn is_float(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Scalar(Scalar::Float(_)))
    }

    pub fn is_char(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Scalar(Scalar::Char))
    }

    pub fn is_int_or_uint(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Scalar(Scalar::Int(_) | Scalar::Uint(_)))
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Scalar(_))
    }

    pub fn is_tuple(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Tuple(..))
    }

    pub fn remove_ref(&self) -> Option<Type<'db>> {
        match &self.ty.kind(Interner) {
            TyKind::Ref(.., ty) => Some(self.derived(ty.clone())),
            _ => None,
        }
    }

    pub fn as_slice(&self) -> Option<Type<'db>> {
        match &self.ty.kind(Interner) {
            TyKind::Slice(ty) => Some(self.derived(ty.clone())),
            _ => None,
        }
    }

    pub fn strip_references(&self) -> Self {
        self.derived(self.ty.strip_references().clone())
    }

    pub fn strip_reference(&self) -> Self {
        self.derived(self.ty.strip_reference().clone())
    }

    pub fn is_unknown(&self) -> bool {
        self.ty.is_unknown()
    }

    /// Checks that particular type `ty` implements `std::future::IntoFuture` or
    /// `std::future::Future` and returns the `Output` associated type.
    /// This function is used in `.await` syntax completion.
    pub fn into_future_output(&self, db: &'db dyn HirDatabase) -> Option<Type<'db>> {
        let trait_ = LangItem::IntoFutureIntoFuture
            .resolve_function(db, self.env.krate)
            .and_then(|into_future_fn| {
                let assoc_item = as_assoc_item(db, AssocItem::Function, into_future_fn)?;
                let into_future_trait = assoc_item.container_or_implemented_trait(db)?;
                Some(into_future_trait.id)
            })
            .or_else(|| LangItem::Future.resolve_trait(db, self.env.krate))?;

        let canonical_ty =
            Canonical { value: self.ty.clone(), binders: CanonicalVarKinds::empty(Interner) };
        if !method_resolution::implements_trait_unique(&canonical_ty, db, &self.env, trait_) {
            return None;
        }

        let output_assoc_type =
            trait_.trait_items(db).associated_type_by_name(&Name::new_symbol_root(sym::Output))?;
        self.normalize_trait_assoc_type(db, &[], output_assoc_type.into())
    }

    /// This does **not** resolve `IntoFuture`, only `Future`.
    pub fn future_output(self, db: &'db dyn HirDatabase) -> Option<Type<'db>> {
        let future_output = LangItem::FutureOutput.resolve_type_alias(db, self.env.krate)?;
        self.normalize_trait_assoc_type(db, &[], future_output.into())
    }

    /// This does **not** resolve `IntoIterator`, only `Iterator`.
    pub fn iterator_item(self, db: &'db dyn HirDatabase) -> Option<Type<'db>> {
        let iterator_trait = LangItem::Iterator.resolve_trait(db, self.env.krate)?;
        let iterator_item = iterator_trait
            .trait_items(db)
            .associated_type_by_name(&Name::new_symbol_root(sym::Item))?;
        self.normalize_trait_assoc_type(db, &[], iterator_item.into())
    }

    pub fn impls_iterator(self, db: &'db dyn HirDatabase) -> bool {
        let Some(iterator_trait) = LangItem::Iterator.resolve_trait(db, self.env.krate) else {
            return false;
        };
        let canonical_ty =
            Canonical { value: self.ty.clone(), binders: CanonicalVarKinds::empty(Interner) };
        method_resolution::implements_trait_unique(&canonical_ty, db, &self.env, iterator_trait)
    }

    /// Resolves the projection `<Self as IntoIterator>::IntoIter` and returns the resulting type
    pub fn into_iterator_iter(self, db: &'db dyn HirDatabase) -> Option<Type<'db>> {
        let trait_ = LangItem::IntoIterIntoIter.resolve_function(db, self.env.krate).and_then(
            |into_iter_fn| {
                let assoc_item = as_assoc_item(db, AssocItem::Function, into_iter_fn)?;
                let into_iter_trait = assoc_item.container_or_implemented_trait(db)?;
                Some(into_iter_trait.id)
            },
        )?;

        let canonical_ty =
            Canonical { value: self.ty.clone(), binders: CanonicalVarKinds::empty(Interner) };
        if !method_resolution::implements_trait_unique(&canonical_ty, db, &self.env, trait_) {
            return None;
        }

        let into_iter_assoc_type = trait_
            .trait_items(db)
            .associated_type_by_name(&Name::new_symbol_root(sym::IntoIter))?;
        self.normalize_trait_assoc_type(db, &[], into_iter_assoc_type.into())
    }

    /// Checks that particular type `ty` implements `std::ops::FnOnce`.
    ///
    /// This function can be used to check if a particular type is callable, since FnOnce is a
    /// supertrait of Fn and FnMut, so all callable types implements at least FnOnce.
    pub fn impls_fnonce(&self, db: &'db dyn HirDatabase) -> bool {
        let fnonce_trait = match FnTrait::FnOnce.get_id(db, self.env.krate) {
            Some(it) => it,
            None => return false,
        };

        let canonical_ty =
            Canonical { value: self.ty.clone(), binders: CanonicalVarKinds::empty(Interner) };
        method_resolution::implements_trait_unique(&canonical_ty, db, &self.env, fnonce_trait)
    }

    // FIXME: Find better API that also handles const generics
    pub fn impls_trait(&self, db: &'db dyn HirDatabase, trait_: Trait, args: &[Type<'db>]) -> bool {
        let mut it = args.iter().map(|t| t.ty.clone());
        let trait_ref = TyBuilder::trait_ref(db, trait_.id)
            .push(self.ty.clone())
            .fill(|x| {
                match x {
                    ParamKind::Type => {
                        it.next().unwrap_or_else(|| TyKind::Error.intern(Interner)).cast(Interner)
                    }
                    ParamKind::Const(ty) => {
                        // FIXME: this code is not covered in tests.
                        unknown_const_as_generic(ty.clone())
                    }
                    ParamKind::Lifetime => error_lifetime().cast(Interner),
                }
            })
            .build();

        let goal = Canonical {
            value: hir_ty::InEnvironment::new(&self.env.env, trait_ref.cast(Interner)),
            binders: CanonicalVarKinds::empty(Interner),
        };

        db.trait_solve(self.env.krate, self.env.block, goal).is_some()
    }

    pub fn normalize_trait_assoc_type(
        &self,
        db: &'db dyn HirDatabase,
        args: &[Type<'db>],
        alias: TypeAlias,
    ) -> Option<Type<'db>> {
        let mut args = args.iter();
        let trait_id = match alias.id.lookup(db).container {
            ItemContainerId::TraitId(id) => id,
            _ => unreachable!("non assoc type alias reached in normalize_trait_assoc_type()"),
        };
        let parent_subst = TyBuilder::subst_for_def(db, trait_id, None)
            .push(self.ty.clone())
            .fill(|it| {
                // FIXME: this code is not covered in tests.
                match it {
                    ParamKind::Type => args.next().unwrap().ty.clone().cast(Interner),
                    ParamKind::Const(ty) => unknown_const_as_generic(ty.clone()),
                    ParamKind::Lifetime => error_lifetime().cast(Interner),
                }
            })
            .build();
        // FIXME: We don't handle GATs yet.
        let projection = TyBuilder::assoc_type_projection(db, alias.id, Some(parent_subst)).build();

        let ty = db.normalize_projection(projection, self.env.clone());
        if ty.is_unknown() { None } else { Some(self.derived(ty)) }
    }

    pub fn is_copy(&self, db: &'db dyn HirDatabase) -> bool {
        let Some(copy_trait) = LangItem::Copy.resolve_trait(db, self.env.krate) else {
            return false;
        };
        self.impls_trait(db, copy_trait.into(), &[])
    }

    pub fn as_callable(&self, db: &'db dyn HirDatabase) -> Option<Callable<'db>> {
        let callee = match self.ty.kind(Interner) {
            TyKind::Closure(id, subst) => Callee::Closure(*id, subst.clone()),
            TyKind::Function(_) => Callee::FnPtr,
            TyKind::FnDef(..) => Callee::Def(self.ty.callable_def(db)?),
            kind => {
                // This will happen when it implements fn or fn mut, since we add an autoborrow adjustment
                let (ty, kind) = if let TyKind::Ref(_, _, ty) = kind {
                    (ty, ty.kind(Interner))
                } else {
                    (&self.ty, kind)
                };
                if let TyKind::Closure(closure, subst) = kind {
                    let sig = ty.callable_sig(db)?;
                    return Some(Callable {
                        ty: self.clone(),
                        sig,
                        callee: Callee::Closure(*closure, subst.clone()),
                        is_bound_method: false,
                    });
                }
                let (fn_trait, sig) = hir_ty::callable_sig_from_fn_trait(ty, self.env.clone(), db)?;
                return Some(Callable {
                    ty: self.clone(),
                    sig,
                    callee: Callee::FnImpl(fn_trait),
                    is_bound_method: false,
                });
            }
        };

        let sig = self.ty.callable_sig(db)?;
        Some(Callable { ty: self.clone(), sig, callee, is_bound_method: false })
    }

    pub fn is_closure(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Closure { .. })
    }

    pub fn as_closure(&self) -> Option<Closure> {
        match self.ty.kind(Interner) {
            TyKind::Closure(id, subst) => Some(Closure { id: *id, subst: subst.clone() }),
            _ => None,
        }
    }

    pub fn is_fn(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::FnDef(..) | TyKind::Function { .. })
    }

    pub fn is_array(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Array(..))
    }

    pub fn is_packed(&self, db: &'db dyn HirDatabase) -> bool {
        let adt_id = match *self.ty.kind(Interner) {
            TyKind::Adt(hir_ty::AdtId(adt_id), ..) => adt_id,
            _ => return false,
        };

        let adt = adt_id.into();
        match adt {
            Adt::Struct(s) => s.repr(db).unwrap_or_default().pack.is_some(),
            _ => false,
        }
    }

    pub fn is_raw_ptr(&self) -> bool {
        matches!(self.ty.kind(Interner), TyKind::Raw(..))
    }

    pub fn remove_raw_ptr(&self) -> Option<Type<'db>> {
        if let TyKind::Raw(_, ty) = self.ty.kind(Interner) {
            Some(self.derived(ty.clone()))
        } else {
            None
        }
    }

    pub fn contains_unknown(&self) -> bool {
        // FIXME: When we get rid of `ConstScalar::Unknown`, we can just look at precomputed
        // `TypeFlags` in `TyData`.
        return go(&self.ty);

        fn go(ty: &Ty) -> bool {
            match ty.kind(Interner) {
                TyKind::Error => true,

                TyKind::Adt(_, substs)
                | TyKind::AssociatedType(_, substs)
                | TyKind::Tuple(_, substs)
                | TyKind::OpaqueType(_, substs)
                | TyKind::FnDef(_, substs)
                | TyKind::Closure(_, substs) => {
                    substs.iter(Interner).filter_map(|a| a.ty(Interner)).any(go)
                }

                TyKind::Array(_ty, len) if len.is_unknown() => true,
                TyKind::Array(ty, _)
                | TyKind::Slice(ty)
                | TyKind::Raw(_, ty)
                | TyKind::Ref(_, _, ty) => go(ty),

                TyKind::Scalar(_)
                | TyKind::Str
                | TyKind::Never
                | TyKind::Placeholder(_)
                | TyKind::BoundVar(_)
                | TyKind::InferenceVar(_, _)
                | TyKind::Dyn(_)
                | TyKind::Function(_)
                | TyKind::Alias(_)
                | TyKind::Foreign(_)
                | TyKind::Coroutine(..)
                | TyKind::CoroutineWitness(..) => false,
            }
        }
    }

    pub fn fields(&self, db: &'db dyn HirDatabase) -> Vec<(Field, Self)> {
        let (variant_id, substs) = match self.ty.kind(Interner) {
            TyKind::Adt(hir_ty::AdtId(AdtId::StructId(s)), substs) => ((*s).into(), substs),
            TyKind::Adt(hir_ty::AdtId(AdtId::UnionId(u)), substs) => ((*u).into(), substs),
            _ => return Vec::new(),
        };

        db.field_types(variant_id)
            .iter()
            .map(|(local_id, ty)| {
                let def = Field { parent: variant_id.into(), id: local_id };
                let ty = ty.clone().substitute(Interner, substs);
                (def, self.derived(ty))
            })
            .collect()
    }

    pub fn tuple_fields(&self, _db: &'db dyn HirDatabase) -> Vec<Self> {
        if let TyKind::Tuple(_, substs) = &self.ty.kind(Interner) {
            substs
                .iter(Interner)
                .map(|ty| self.derived(ty.assert_ty_ref(Interner).clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn as_array(&self, db: &'db dyn HirDatabase) -> Option<(Self, usize)> {
        if let TyKind::Array(ty, len) = &self.ty.kind(Interner) {
            try_const_usize(db, len).map(|it| (self.derived(ty.clone()), it as usize))
        } else {
            None
        }
    }

    pub fn fingerprint_for_trait_impl(&self) -> Option<TyFingerprint> {
        TyFingerprint::for_trait_impl(&self.ty)
    }

    pub(crate) fn canonical(&self) -> Canonical<Ty> {
        hir_ty::replace_errors_with_variables(&self.ty)
    }

    /// Returns types that this type dereferences to (including this type itself). The returned
    /// iterator won't yield the same type more than once even if the deref chain contains a cycle.
    pub fn autoderef(
        &self,
        db: &'db dyn HirDatabase,
    ) -> impl Iterator<Item = Type<'db>> + use<'_, 'db> {
        self.autoderef_(db).map(move |ty| self.derived(ty))
    }

    fn autoderef_(&self, db: &'db dyn HirDatabase) -> impl Iterator<Item = Ty> {
        // There should be no inference vars in types passed here
        let canonical = hir_ty::replace_errors_with_variables(&self.ty);
        autoderef(db, self.env.clone(), canonical)
    }

    // This would be nicer if it just returned an iterator, but that runs into
    // lifetime problems, because we need to borrow temp `CrateImplDefs`.
    pub fn iterate_assoc_items<T>(
        &self,
        db: &'db dyn HirDatabase,
        krate: Crate,
        mut callback: impl FnMut(AssocItem) -> Option<T>,
    ) -> Option<T> {
        let mut slot = None;
        self.iterate_assoc_items_dyn(db, krate, &mut |assoc_item_id| {
            slot = callback(assoc_item_id.into());
            slot.is_some()
        });
        slot
    }

    fn iterate_assoc_items_dyn(
        &self,
        db: &'db dyn HirDatabase,
        krate: Crate,
        callback: &mut dyn FnMut(AssocItemId) -> bool,
    ) {
        let def_crates = match method_resolution::def_crates(db, &self.ty, krate.id) {
            Some(it) => it,
            None => return,
        };
        for krate in def_crates {
            let impls = db.inherent_impls_in_crate(krate);

            for impl_def in impls.for_self_ty(&self.ty) {
                for &(_, item) in impl_def.impl_items(db).items.iter() {
                    if callback(item) {
                        return;
                    }
                }
            }
        }
    }

    /// Iterates its type arguments
    ///
    /// It iterates the actual type arguments when concrete types are used
    /// and otherwise the generic names.
    /// It does not include `const` arguments.
    ///
    /// For code, such as:
    /// ```text
    /// struct Foo<T, U>
    ///
    /// impl<U> Foo<String, U>
    /// ```
    ///
    /// It iterates:
    /// ```text
    /// - "String"
    /// - "U"
    /// ```
    pub fn type_arguments(&self) -> impl Iterator<Item = Type<'db>> + '_ {
        self.ty
            .strip_references()
            .as_adt()
            .map(|(_, substs)| substs)
            .or_else(|| self.ty.strip_references().as_tuple())
            .into_iter()
            .flat_map(|substs| substs.iter(Interner))
            .filter_map(|arg| arg.ty(Interner).cloned())
            .map(move |ty| self.derived(ty))
    }

    /// Iterates its type and const arguments
    ///
    /// It iterates the actual type and const arguments when concrete types
    /// are used and otherwise the generic names.
    ///
    /// For code, such as:
    /// ```text
    /// struct Foo<T, const U: usize, const X: usize>
    ///
    /// impl<U> Foo<String, U, 12>
    /// ```
    ///
    /// It iterates:
    /// ```text
    /// - "String"
    /// - "U"
    /// - "12"
    /// ```
    pub fn type_and_const_arguments<'a>(
        &'a self,
        db: &'a dyn HirDatabase,
        display_target: DisplayTarget,
    ) -> impl Iterator<Item = SmolStr> + 'a {
        self.ty
            .strip_references()
            .as_adt()
            .into_iter()
            .flat_map(|(_, substs)| substs.iter(Interner))
            .filter_map(move |arg| {
                // arg can be either a `Ty` or `constant`
                if let Some(ty) = arg.ty(Interner) {
                    Some(format_smolstr!("{}", ty.display(db, display_target)))
                } else {
                    arg.constant(Interner)
                        .map(|const_| format_smolstr!("{}", const_.display(db, display_target)))
                }
            })
    }

    /// Combines lifetime indicators, type and constant parameters into a single `Iterator`
    pub fn generic_parameters<'a>(
        &'a self,
        db: &'a dyn HirDatabase,
        display_target: DisplayTarget,
    ) -> impl Iterator<Item = SmolStr> + 'a {
        // iterate the lifetime
        self.as_adt()
            .and_then(|a| {
                // Lifetimes do not need edition-specific handling as they cannot be escaped.
                a.lifetime(db).map(|lt| lt.name.display_no_db(Edition::Edition2015).to_smolstr())
            })
            .into_iter()
            // add the type and const parameters
            .chain(self.type_and_const_arguments(db, display_target))
    }

    pub fn iterate_method_candidates_with_traits<T>(
        &self,
        db: &'db dyn HirDatabase,
        scope: &SemanticsScope<'_>,
        traits_in_scope: &FxHashSet<TraitId>,
        with_local_impls: Option<Module>,
        name: Option<&Name>,
        mut callback: impl FnMut(Function) -> Option<T>,
    ) -> Option<T> {
        let _p = tracing::info_span!("iterate_method_candidates_with_traits").entered();
        let mut slot = None;
        self.iterate_method_candidates_split_inherent(
            db,
            scope,
            traits_in_scope,
            with_local_impls,
            name,
            |f| match callback(f) {
                it @ Some(_) => {
                    slot = it;
                    ControlFlow::Break(())
                }
                None => ControlFlow::Continue(()),
            },
        );
        slot
    }

    pub fn iterate_method_candidates<T>(
        &self,
        db: &'db dyn HirDatabase,
        scope: &SemanticsScope<'_>,
        with_local_impls: Option<Module>,
        name: Option<&Name>,
        callback: impl FnMut(Function) -> Option<T>,
    ) -> Option<T> {
        self.iterate_method_candidates_with_traits(
            db,
            scope,
            &scope.visible_traits().0,
            with_local_impls,
            name,
            callback,
        )
    }

    /// Allows you to treat inherent and non-inherent methods differently.
    ///
    /// Note that inherent methods may actually be trait methods! For example, in `dyn Trait`, the trait's methods
    /// are considered inherent methods.
    pub fn iterate_method_candidates_split_inherent(
        &self,
        db: &'db dyn HirDatabase,
        scope: &SemanticsScope<'_>,
        traits_in_scope: &FxHashSet<TraitId>,
        with_local_impls: Option<Module>,
        name: Option<&Name>,
        callback: impl MethodCandidateCallback,
    ) {
        struct Callback<T>(T);

        impl<T: MethodCandidateCallback> method_resolution::MethodCandidateCallback for Callback<T> {
            fn on_inherent_method(
                &mut self,
                _adjustments: method_resolution::ReceiverAdjustments,
                item: AssocItemId,
                _is_visible: bool,
            ) -> ControlFlow<()> {
                if let AssocItemId::FunctionId(func) = item {
                    self.0.on_inherent_method(func.into())
                } else {
                    ControlFlow::Continue(())
                }
            }

            fn on_trait_method(
                &mut self,
                _adjustments: method_resolution::ReceiverAdjustments,
                item: AssocItemId,
                _is_visible: bool,
            ) -> ControlFlow<()> {
                if let AssocItemId::FunctionId(func) = item {
                    self.0.on_trait_method(func.into())
                } else {
                    ControlFlow::Continue(())
                }
            }
        }

        let _p = tracing::info_span!(
            "iterate_method_candidates_dyn",
            with_local_impls = traits_in_scope.len(),
            traits_in_scope = traits_in_scope.len(),
            ?name,
        )
        .entered();
        // There should be no inference vars in types passed here
        let canonical = hir_ty::replace_errors_with_variables(&self.ty);

        let krate = scope.krate();
        let environment = scope
            .resolver()
            .generic_def()
            .map_or_else(|| TraitEnvironment::empty(krate.id), |d| db.trait_environment(d));

        _ = method_resolution::iterate_method_candidates_dyn(
            &canonical,
            db,
            environment,
            traits_in_scope,
            with_local_impls.and_then(|b| b.id.containing_block()).into(),
            name,
            method_resolution::LookupMode::MethodCall,
            &mut Callback(callback),
        );
    }

    #[tracing::instrument(skip_all, fields(name = ?name))]
    pub fn iterate_path_candidates<T>(
        &self,
        db: &'db dyn HirDatabase,
        scope: &SemanticsScope<'_>,
        traits_in_scope: &FxHashSet<TraitId>,
        with_local_impls: Option<Module>,
        name: Option<&Name>,
        mut callback: impl FnMut(AssocItem) -> Option<T>,
    ) -> Option<T> {
        let _p = tracing::info_span!("iterate_path_candidates").entered();
        let mut slot = None;

        self.iterate_path_candidates_split_inherent(
            db,
            scope,
            traits_in_scope,
            with_local_impls,
            name,
            |item| match callback(item) {
                it @ Some(_) => {
                    slot = it;
                    ControlFlow::Break(())
                }
                None => ControlFlow::Continue(()),
            },
        );
        slot
    }

    /// Iterates over inherent methods.
    ///
    /// In some circumstances, inherent methods methods may actually be trait methods!
    /// For example, when `dyn Trait` is a receiver, _trait_'s methods would be considered
    /// to be inherent methods.
    #[tracing::instrument(skip_all, fields(name = ?name))]
    pub fn iterate_path_candidates_split_inherent(
        &self,
        db: &'db dyn HirDatabase,
        scope: &SemanticsScope<'_>,
        traits_in_scope: &FxHashSet<TraitId>,
        with_local_impls: Option<Module>,
        name: Option<&Name>,
        callback: impl PathCandidateCallback,
    ) {
        struct Callback<T>(T);

        impl<T: PathCandidateCallback> method_resolution::MethodCandidateCallback for Callback<T> {
            fn on_inherent_method(
                &mut self,
                _adjustments: method_resolution::ReceiverAdjustments,
                item: AssocItemId,
                _is_visible: bool,
            ) -> ControlFlow<()> {
                self.0.on_inherent_item(item.into())
            }

            fn on_trait_method(
                &mut self,
                _adjustments: method_resolution::ReceiverAdjustments,
                item: AssocItemId,
                _is_visible: bool,
            ) -> ControlFlow<()> {
                self.0.on_trait_item(item.into())
            }
        }

        let canonical = hir_ty::replace_errors_with_variables(&self.ty);

        let krate = scope.krate();
        let environment = scope
            .resolver()
            .generic_def()
            .map_or_else(|| TraitEnvironment::empty(krate.id), |d| db.trait_environment(d));

        _ = method_resolution::iterate_path_candidates(
            &canonical,
            db,
            environment,
            traits_in_scope,
            with_local_impls.and_then(|b| b.id.containing_block()).into(),
            name,
            &mut Callback(callback),
        );
    }

    pub fn as_adt(&self) -> Option<Adt> {
        let (adt, _subst) = self.ty.as_adt()?;
        Some(adt.into())
    }

    pub fn as_builtin(&self) -> Option<BuiltinType> {
        self.ty.as_builtin().map(|inner| BuiltinType { inner })
    }

    pub fn as_dyn_trait(&self) -> Option<Trait> {
        self.ty.dyn_trait().map(Into::into)
    }

    /// If a type can be represented as `dyn Trait`, returns all traits accessible via this type,
    /// or an empty iterator otherwise.
    pub fn applicable_inherent_traits(
        &self,
        db: &'db dyn HirDatabase,
    ) -> impl Iterator<Item = Trait> {
        let _p = tracing::info_span!("applicable_inherent_traits").entered();
        self.autoderef_(db)
            .filter_map(|ty| ty.dyn_trait())
            .flat_map(move |dyn_trait_id| hir_ty::all_super_traits(db, dyn_trait_id))
            .map(Trait::from)
    }

    pub fn env_traits(&self, db: &'db dyn HirDatabase) -> impl Iterator<Item = Trait> {
        let _p = tracing::info_span!("env_traits").entered();
        self.autoderef_(db)
            .filter(|ty| matches!(ty.kind(Interner), TyKind::Placeholder(_)))
            .flat_map(|ty| {
                self.env
                    .traits_in_scope_from_clauses(ty)
                    .flat_map(|t| hir_ty::all_super_traits(db, t))
            })
            .map(Trait::from)
    }

    pub fn as_impl_traits(&self, db: &'db dyn HirDatabase) -> Option<impl Iterator<Item = Trait>> {
        self.ty.impl_trait_bounds(db).map(|it| {
            it.into_iter().filter_map(|pred| match pred.skip_binders() {
                hir_ty::WhereClause::Implemented(trait_ref) => {
                    Some(Trait::from(trait_ref.hir_trait_id()))
                }
                _ => None,
            })
        })
    }

    pub fn as_associated_type_parent_trait(&self, db: &'db dyn HirDatabase) -> Option<Trait> {
        self.ty.associated_type_parent_trait(db).map(Into::into)
    }

    fn derived(&self, ty: Ty) -> Self {
        Type { env: self.env.clone(), ty, _pd: PhantomCovariantLifetime::new() }
    }

    /// Visits every type, including generic arguments, in this type. `cb` is called with type
    /// itself first, and then with its generic arguments.
    pub fn walk(&self, db: &'db dyn HirDatabase, mut cb: impl FnMut(Type<'db>)) {
        fn walk_substs<'db>(
            db: &'db dyn HirDatabase,
            type_: &Type<'db>,
            substs: &Substitution,
            cb: &mut impl FnMut(Type<'db>),
        ) {
            for ty in substs.iter(Interner).filter_map(|a| a.ty(Interner)) {
                walk_type(db, &type_.derived(ty.clone()), cb);
            }
        }

        fn walk_bounds<'db>(
            db: &'db dyn HirDatabase,
            type_: &Type<'db>,
            bounds: &[QuantifiedWhereClause],
            cb: &mut impl FnMut(Type<'db>),
        ) {
            for pred in bounds {
                if let WhereClause::Implemented(trait_ref) = pred.skip_binders() {
                    cb(type_.clone());
                    // skip the self type. it's likely the type we just got the bounds from
                    if let [self_ty, params @ ..] = trait_ref.substitution.as_slice(Interner) {
                        for ty in
                            params.iter().filter(|&ty| ty != self_ty).filter_map(|a| a.ty(Interner))
                        {
                            walk_type(db, &type_.derived(ty.clone()), cb);
                        }
                    }
                }
            }
        }

        fn walk_type<'db>(
            db: &'db dyn HirDatabase,
            type_: &Type<'db>,
            cb: &mut impl FnMut(Type<'db>),
        ) {
            let ty = type_.ty.strip_references();
            match ty.kind(Interner) {
                TyKind::Adt(_, substs) => {
                    cb(type_.derived(ty.clone()));
                    walk_substs(db, type_, substs, cb);
                }
                TyKind::AssociatedType(_, substs) => {
                    if ty.associated_type_parent_trait(db).is_some() {
                        cb(type_.derived(ty.clone()));
                    }
                    walk_substs(db, type_, substs, cb);
                }
                TyKind::OpaqueType(_, subst) => {
                    if let Some(bounds) = ty.impl_trait_bounds(db) {
                        walk_bounds(db, &type_.derived(ty.clone()), &bounds, cb);
                    }

                    walk_substs(db, type_, subst, cb);
                }
                TyKind::Alias(AliasTy::Opaque(opaque_ty)) => {
                    if let Some(bounds) = ty.impl_trait_bounds(db) {
                        walk_bounds(db, &type_.derived(ty.clone()), &bounds, cb);
                    }

                    walk_substs(db, type_, &opaque_ty.substitution, cb);
                }
                TyKind::Placeholder(_) => {
                    cb(type_.derived(ty.clone()));
                    if let Some(bounds) = ty.impl_trait_bounds(db) {
                        walk_bounds(db, &type_.derived(ty.clone()), &bounds, cb);
                    }
                }
                TyKind::Dyn(bounds) => {
                    walk_bounds(
                        db,
                        &type_.derived(ty.clone()),
                        bounds.bounds.skip_binders().interned(),
                        cb,
                    );
                }

                TyKind::Ref(_, _, ty)
                | TyKind::Raw(_, ty)
                | TyKind::Array(ty, _)
                | TyKind::Slice(ty) => {
                    walk_type(db, &type_.derived(ty.clone()), cb);
                }

                TyKind::FnDef(_, substs)
                | TyKind::Tuple(_, substs)
                | TyKind::Closure(.., substs) => {
                    walk_substs(db, type_, substs, cb);
                }
                TyKind::Function(hir_ty::FnPointer { substitution, .. }) => {
                    walk_substs(db, type_, &substitution.0, cb);
                }

                _ => {}
            }
        }

        walk_type(db, self, &mut cb);
    }
    /// Check if type unifies with another type.
    ///
    /// Note that we consider placeholder types to unify with everything.
    /// For example `Option<T>` and `Option<U>` unify although there is unresolved goal `T = U`.
    pub fn could_unify_with(&self, db: &'db dyn HirDatabase, other: &Type<'db>) -> bool {
        let tys = hir_ty::replace_errors_with_variables(&(self.ty.clone(), other.ty.clone()));
        hir_ty::could_unify(db, self.env.clone(), &tys)
    }

    /// Check if type unifies with another type eagerly making sure there are no unresolved goals.
    ///
    /// This means that placeholder types are not considered to unify if there are any bounds set on
    /// them. For example `Option<T>` and `Option<U>` do not unify as we cannot show that `T = U`
    pub fn could_unify_with_deeply(&self, db: &'db dyn HirDatabase, other: &Type<'db>) -> bool {
        let tys = hir_ty::replace_errors_with_variables(&(self.ty.clone(), other.ty.clone()));
        hir_ty::could_unify_deeply(db, self.env.clone(), &tys)
    }

    pub fn could_coerce_to(&self, db: &'db dyn HirDatabase, to: &Type<'db>) -> bool {
        let tys = hir_ty::replace_errors_with_variables(&(self.ty.clone(), to.ty.clone()));
        hir_ty::could_coerce(db, self.env.clone(), &tys)
    }

    pub fn as_type_param(&self, db: &'db dyn HirDatabase) -> Option<TypeParam> {
        match self.ty.kind(Interner) {
            TyKind::Placeholder(p) => Some(TypeParam {
                id: TypeParamId::from_unchecked(hir_ty::from_placeholder_idx(db, *p)),
            }),
            _ => None,
        }
    }

    /// Returns unique `GenericParam`s contained in this type.
    pub fn generic_params(&self, db: &'db dyn HirDatabase) -> FxHashSet<GenericParam> {
        hir_ty::collect_placeholders(&self.ty, db)
            .into_iter()
            .map(|id| TypeOrConstParam { id }.split(db).either_into())
            .collect()
    }

    pub fn layout(&self, db: &'db dyn HirDatabase) -> Result<Layout, LayoutError> {
        db.layout_of_ty(self.ty.clone(), self.env.clone())
            .map(|layout| Layout(layout, db.target_data_layout(self.env.krate).unwrap()))
    }

    pub fn drop_glue(&self, db: &'db dyn HirDatabase) -> DropGlue {
        db.has_drop_glue(self.ty.clone(), self.env.clone())
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub struct InlineAsmOperand {
    owner: DefWithBodyId,
    expr: ExprId,
    index: usize,
}

impl InlineAsmOperand {
    pub fn parent(self, _db: &dyn HirDatabase) -> DefWithBody {
        self.owner.into()
    }

    pub fn name(&self, db: &dyn HirDatabase) -> Option<Name> {
        match &db.body(self.owner)[self.expr] {
            hir_def::hir::Expr::InlineAsm(e) => e.operands.get(self.index)?.0.clone(),
            _ => None,
        }
    }
}

// FIXME: Document this
#[derive(Debug)]
pub struct Callable<'db> {
    ty: Type<'db>,
    sig: CallableSig,
    callee: Callee,
    /// Whether this is a method that was called with method call syntax.
    is_bound_method: bool,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Callee {
    Def(CallableDefId),
    Closure(ClosureId, Substitution),
    FnPtr,
    FnImpl(FnTrait),
}

pub enum CallableKind {
    Function(Function),
    TupleStruct(Struct),
    TupleEnumVariant(Variant),
    Closure(Closure),
    FnPtr,
    FnImpl(FnTrait),
}

impl<'db> Callable<'db> {
    pub fn kind(&self) -> CallableKind {
        match self.callee {
            Callee::Def(CallableDefId::FunctionId(it)) => CallableKind::Function(it.into()),
            Callee::Def(CallableDefId::StructId(it)) => CallableKind::TupleStruct(it.into()),
            Callee::Def(CallableDefId::EnumVariantId(it)) => {
                CallableKind::TupleEnumVariant(it.into())
            }
            Callee::Closure(id, ref subst) => {
                CallableKind::Closure(Closure { id, subst: subst.clone() })
            }
            Callee::FnPtr => CallableKind::FnPtr,
            Callee::FnImpl(fn_) => CallableKind::FnImpl(fn_),
        }
    }
    pub fn receiver_param(&self, db: &'db dyn HirDatabase) -> Option<(SelfParam, Type<'db>)> {
        let func = match self.callee {
            Callee::Def(CallableDefId::FunctionId(it)) if self.is_bound_method => it,
            _ => return None,
        };
        let func = Function { id: func };
        Some((func.self_param(db)?, self.ty.derived(self.sig.params()[0].clone())))
    }
    pub fn n_params(&self) -> usize {
        self.sig.params().len() - if self.is_bound_method { 1 } else { 0 }
    }
    pub fn params(&self) -> Vec<Param<'db>> {
        self.sig
            .params()
            .iter()
            .enumerate()
            .skip(if self.is_bound_method { 1 } else { 0 })
            .map(|(idx, ty)| (idx, self.ty.derived(ty.clone())))
            .map(|(idx, ty)| Param { func: self.callee.clone(), idx, ty })
            .collect()
    }
    pub fn return_type(&self) -> Type<'db> {
        self.ty.derived(self.sig.ret().clone())
    }
    pub fn sig(&self) -> &CallableSig {
        &self.sig
    }

    pub fn ty(&self) -> &Type<'db> {
        &self.ty
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Layout(Arc<TyLayout>, Arc<TargetDataLayout>);

impl Layout {
    pub fn size(&self) -> u64 {
        self.0.size.bytes()
    }

    pub fn align(&self) -> u64 {
        self.0.align.abi.bytes()
    }

    pub fn niches(&self) -> Option<u128> {
        Some(self.0.largest_niche?.available(&*self.1))
    }

    pub fn field_offset(&self, field: Field) -> Option<u64> {
        match self.0.fields {
            layout::FieldsShape::Primitive => None,
            layout::FieldsShape::Union(_) => Some(0),
            layout::FieldsShape::Array { stride, count } => {
                let i = u64::try_from(field.index()).ok()?;
                (i < count).then_some((stride * i).bytes())
            }
            layout::FieldsShape::Arbitrary { ref offsets, .. } => {
                Some(offsets.get(RustcFieldIdx(field.id))?.bytes())
            }
        }
    }

    pub fn tuple_field_offset(&self, field: usize) -> Option<u64> {
        match self.0.fields {
            layout::FieldsShape::Primitive => None,
            layout::FieldsShape::Union(_) => Some(0),
            layout::FieldsShape::Array { stride, count } => {
                let i = u64::try_from(field).ok()?;
                (i < count).then_some((stride * i).bytes())
            }
            layout::FieldsShape::Arbitrary { ref offsets, .. } => {
                Some(offsets.get(RustcFieldIdx::new(field))?.bytes())
            }
        }
    }

    pub fn tail_padding(&self, field_size: &mut impl FnMut(usize) -> Option<u64>) -> Option<u64> {
        match self.0.fields {
            layout::FieldsShape::Primitive => None,
            layout::FieldsShape::Union(_) => None,
            layout::FieldsShape::Array { stride, count } => count.checked_sub(1).and_then(|tail| {
                let tail_field_size = field_size(tail as usize)?;
                let offset = stride.bytes() * tail;
                self.0.size.bytes().checked_sub(offset)?.checked_sub(tail_field_size)
            }),
            layout::FieldsShape::Arbitrary { ref offsets, ref memory_index } => {
                let tail = memory_index.last_index()?;
                let tail_field_size = field_size(tail.0.into_raw().into_u32() as usize)?;
                let offset = offsets.get(tail)?.bytes();
                self.0.size.bytes().checked_sub(offset)?.checked_sub(tail_field_size)
            }
        }
    }

    pub fn largest_padding(
        &self,
        field_size: &mut impl FnMut(usize) -> Option<u64>,
    ) -> Option<u64> {
        match self.0.fields {
            layout::FieldsShape::Primitive => None,
            layout::FieldsShape::Union(_) => None,
            layout::FieldsShape::Array { stride: _, count: 0 } => None,
            layout::FieldsShape::Array { stride, .. } => {
                let size = field_size(0)?;
                stride.bytes().checked_sub(size)
            }
            layout::FieldsShape::Arbitrary { ref offsets, ref memory_index } => {
                let mut reverse_index = vec![None; memory_index.len()];
                for (src, (mem, offset)) in memory_index.iter().zip(offsets.iter()).enumerate() {
                    reverse_index[*mem as usize] = Some((src, offset.bytes()));
                }
                if reverse_index.iter().any(|it| it.is_none()) {
                    stdx::never!();
                    return None;
                }
                reverse_index
                    .into_iter()
                    .flatten()
                    .chain(std::iter::once((0, self.0.size.bytes())))
                    .tuple_windows()
                    .filter_map(|((i, start), (_, end))| {
                        let size = field_size(i)?;
                        end.checked_sub(start)?.checked_sub(size)
                    })
                    .max()
            }
        }
    }

    pub fn enum_tag_size(&self) -> Option<usize> {
        let tag_size =
            if let layout::Variants::Multiple { tag, tag_encoding, .. } = &self.0.variants {
                match tag_encoding {
                    TagEncoding::Direct => tag.size(&*self.1).bytes_usize(),
                    TagEncoding::Niche { .. } => 0,
                }
            } else {
                return None;
            };
        Some(tag_size)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BindingMode {
    Move,
    Ref(Mutability),
}

/// For IDE only
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScopeDef {
    ModuleDef(ModuleDef),
    GenericParam(GenericParam),
    ImplSelfType(Impl),
    AdtSelfType(Adt),
    Local(Local),
    Label(Label),
    Unknown,
}

impl ScopeDef {
    pub fn all_items(def: PerNs) -> ArrayVec<Self, 3> {
        let mut items = ArrayVec::new();

        match (def.take_types(), def.take_values()) {
            (Some(m1), None) => items.push(ScopeDef::ModuleDef(m1.into())),
            (None, Some(m2)) => items.push(ScopeDef::ModuleDef(m2.into())),
            (Some(m1), Some(m2)) => {
                // Some items, like unit structs and enum variants, are
                // returned as both a type and a value. Here we want
                // to de-duplicate them.
                if m1 != m2 {
                    items.push(ScopeDef::ModuleDef(m1.into()));
                    items.push(ScopeDef::ModuleDef(m2.into()));
                } else {
                    items.push(ScopeDef::ModuleDef(m1.into()));
                }
            }
            (None, None) => {}
        };

        if let Some(macro_def_id) = def.take_macros() {
            items.push(ScopeDef::ModuleDef(ModuleDef::Macro(macro_def_id.into())));
        }

        if items.is_empty() {
            items.push(ScopeDef::Unknown);
        }

        items
    }

    pub fn attrs(&self, db: &dyn HirDatabase) -> Option<AttrsWithOwner> {
        match self {
            ScopeDef::ModuleDef(it) => it.attrs(db),
            ScopeDef::GenericParam(it) => Some(it.attrs(db)),
            ScopeDef::ImplSelfType(_)
            | ScopeDef::AdtSelfType(_)
            | ScopeDef::Local(_)
            | ScopeDef::Label(_)
            | ScopeDef::Unknown => None,
        }
    }

    pub fn krate(&self, db: &dyn HirDatabase) -> Option<Crate> {
        match self {
            ScopeDef::ModuleDef(it) => it.module(db).map(|m| m.krate()),
            ScopeDef::GenericParam(it) => Some(it.module(db).krate()),
            ScopeDef::ImplSelfType(_) => None,
            ScopeDef::AdtSelfType(it) => Some(it.module(db).krate()),
            ScopeDef::Local(it) => Some(it.module(db).krate()),
            ScopeDef::Label(it) => Some(it.module(db).krate()),
            ScopeDef::Unknown => None,
        }
    }
}

impl From<ItemInNs> for ScopeDef {
    fn from(item: ItemInNs) -> Self {
        match item {
            ItemInNs::Types(id) => ScopeDef::ModuleDef(id),
            ItemInNs::Values(id) => ScopeDef::ModuleDef(id),
            ItemInNs::Macros(id) => ScopeDef::ModuleDef(ModuleDef::Macro(id)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Adjustment<'db> {
    pub source: Type<'db>,
    pub target: Type<'db>,
    pub kind: Adjust,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Adjust {
    /// Go from ! to any type.
    NeverToAny,
    /// Dereference once, producing a place.
    Deref(Option<OverloadedDeref>),
    /// Take the address and produce either a `&` or `*` pointer.
    Borrow(AutoBorrow),
    Pointer(PointerCast),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AutoBorrow {
    /// Converts from T to &T.
    Ref(Mutability),
    /// Converts from T to *T.
    RawPtr(Mutability),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OverloadedDeref(pub Mutability);

pub trait HasVisibility {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility;
    fn is_visible_from(&self, db: &dyn HirDatabase, module: Module) -> bool {
        let vis = self.visibility(db);
        vis.is_visible_from(db, module.id)
    }
}

/// Trait for obtaining the defining crate of an item.
pub trait HasCrate {
    fn krate(&self, db: &dyn HirDatabase) -> Crate;
}

impl<T: hir_def::HasModule> HasCrate for T {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate().into()
    }
}

impl HasCrate for AssocItem {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for Struct {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for Union {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for Enum {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for Field {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.parent_def(db).module(db).krate()
    }
}

impl HasCrate for Variant {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for Function {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for Const {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for TypeAlias {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for Type<'_> {
    fn krate(&self, _db: &dyn HirDatabase) -> Crate {
        self.env.krate.into()
    }
}

impl HasCrate for Macro {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for Trait {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for TraitAlias {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for Static {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for Adt {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for Impl {
    fn krate(&self, db: &dyn HirDatabase) -> Crate {
        self.module(db).krate()
    }
}

impl HasCrate for Module {
    fn krate(&self, _: &dyn HirDatabase) -> Crate {
        Module::krate(*self)
    }
}

pub trait HasContainer {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer;
}

impl HasContainer for ExternCrateDecl {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer {
        container_id_to_hir(self.id.lookup(db).container.into())
    }
}

impl HasContainer for Module {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer {
        // FIXME: handle block expressions as modules (their parent is in a different DefMap)
        let def_map = self.id.def_map(db);
        match def_map[self.id.local_id].parent {
            Some(parent_id) => ItemContainer::Module(Module { id: def_map.module_id(parent_id) }),
            None => ItemContainer::Crate(def_map.krate().into()),
        }
    }
}

impl HasContainer for Function {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer {
        container_id_to_hir(self.id.lookup(db).container)
    }
}

impl HasContainer for Struct {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer {
        ItemContainer::Module(Module { id: self.id.lookup(db).container })
    }
}

impl HasContainer for Union {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer {
        ItemContainer::Module(Module { id: self.id.lookup(db).container })
    }
}

impl HasContainer for Enum {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer {
        ItemContainer::Module(Module { id: self.id.lookup(db).container })
    }
}

impl HasContainer for TypeAlias {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer {
        container_id_to_hir(self.id.lookup(db).container)
    }
}

impl HasContainer for Const {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer {
        container_id_to_hir(self.id.lookup(db).container)
    }
}

impl HasContainer for Static {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer {
        container_id_to_hir(self.id.lookup(db).container)
    }
}

impl HasContainer for Trait {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer {
        ItemContainer::Module(Module { id: self.id.lookup(db).container })
    }
}

impl HasContainer for TraitAlias {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer {
        ItemContainer::Module(Module { id: self.id.lookup(db).container })
    }
}

impl HasContainer for ExternBlock {
    fn container(&self, db: &dyn HirDatabase) -> ItemContainer {
        ItemContainer::Module(Module { id: self.id.lookup(db).container })
    }
}

fn container_id_to_hir(c: ItemContainerId) -> ItemContainer {
    match c {
        ItemContainerId::ExternBlockId(id) => ItemContainer::ExternBlock(ExternBlock { id }),
        ItemContainerId::ModuleId(id) => ItemContainer::Module(Module { id }),
        ItemContainerId::ImplId(id) => ItemContainer::Impl(Impl { id }),
        ItemContainerId::TraitId(id) => ItemContainer::Trait(Trait { id }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemContainer {
    Trait(Trait),
    Impl(Impl),
    Module(Module),
    ExternBlock(ExternBlock),
    Crate(Crate),
}

/// Subset of `ide_db::Definition` that doc links can resolve to.
pub enum DocLinkDef {
    ModuleDef(ModuleDef),
    Field(Field),
    SelfType(Trait),
}

fn push_ty_diagnostics<'db>(
    db: &'db dyn HirDatabase,
    acc: &mut Vec<AnyDiagnostic<'db>>,
    diagnostics: Option<ThinArc<(), TyLoweringDiagnostic>>,
    source_map: &ExpressionStoreSourceMap,
) {
    if let Some(diagnostics) = diagnostics {
        acc.extend(
            diagnostics
                .slice
                .iter()
                .filter_map(|diagnostic| AnyDiagnostic::ty_diagnostic(diagnostic, source_map, db)),
        );
    }
}

pub trait MethodCandidateCallback {
    fn on_inherent_method(&mut self, f: Function) -> ControlFlow<()>;

    fn on_trait_method(&mut self, f: Function) -> ControlFlow<()>;
}

impl<F> MethodCandidateCallback for F
where
    F: FnMut(Function) -> ControlFlow<()>,
{
    fn on_inherent_method(&mut self, f: Function) -> ControlFlow<()> {
        self(f)
    }

    fn on_trait_method(&mut self, f: Function) -> ControlFlow<()> {
        self(f)
    }
}

pub trait PathCandidateCallback {
    fn on_inherent_item(&mut self, item: AssocItem) -> ControlFlow<()>;

    fn on_trait_item(&mut self, item: AssocItem) -> ControlFlow<()>;
}

impl<F> PathCandidateCallback for F
where
    F: FnMut(AssocItem) -> ControlFlow<()>,
{
    fn on_inherent_item(&mut self, item: AssocItem) -> ControlFlow<()> {
        self(item)
    }

    fn on_trait_item(&mut self, item: AssocItem) -> ControlFlow<()> {
        self(item)
    }
}

pub fn resolve_absolute_path<'a, I: Iterator<Item = Symbol> + Clone + 'a>(
    db: &'a dyn HirDatabase,
    mut segments: I,
) -> impl Iterator<Item = ItemInNs> + use<'a, I> {
    segments
        .next()
        .into_iter()
        .flat_map(move |crate_name| {
            db.all_crates()
                .iter()
                .filter(|&krate| {
                    krate
                        .extra_data(db)
                        .display_name
                        .as_ref()
                        .is_some_and(|name| *name.crate_name().symbol() == crate_name)
                })
                .filter_map(|&krate| {
                    let segments = segments.clone();
                    let mut def_map = crate_def_map(db, krate);
                    let mut module = &def_map[DefMap::ROOT];
                    let mut segments = segments.with_position().peekable();
                    while let Some((_, segment)) = segments.next_if(|&(position, _)| {
                        !matches!(position, itertools::Position::Last | itertools::Position::Only)
                    }) {
                        let res = module
                            .scope
                            .get(&Name::new_symbol_root(segment))
                            .take_types()
                            .and_then(|res| match res {
                                ModuleDefId::ModuleId(it) => Some(it),
                                _ => None,
                            })?;
                        def_map = res.def_map(db);
                        module = &def_map[res.local_id];
                    }
                    let (_, item_name) = segments.next()?;
                    let res = module.scope.get(&Name::new_symbol_root(item_name));
                    Some(res.iter_items().map(|(item, _)| item.into()))
                })
                .collect::<Vec<_>>()
        })
        .flatten()
}

fn as_name_opt(name: Option<impl AsName>) -> Name {
    name.map_or_else(Name::missing, |name| name.as_name())
}
