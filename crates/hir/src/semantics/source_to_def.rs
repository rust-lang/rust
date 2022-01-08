//! Maps *syntax* of various definitions to their semantic ids.
//!
//! This is a very interesting module, and, in some sense, can be considered the
//! heart of the IDE parts of rust-analyzer.
//!
//! This module solves the following problem:
//!
//!     Given a piece of syntax, find the corresponding semantic definition (def).
//!
//! This problem is a part of more-or-less every IDE feature implemented. Every
//! IDE functionality (like goto to definition), conceptually starts with a
//! specific cursor position in a file. Starting with this text offset, we first
//! figure out what syntactic construct are we at: is this a pattern, an
//! expression, an item definition.
//!
//! Knowing only the syntax gives us relatively little info. For example,
//! looking at the syntax of the function we can realise that it is a part of an
//! `impl` block, but we won't be able to tell what trait function the current
//! function overrides, and whether it does that correctly. For that, we need to
//! go from [`ast::Fn`] to [`crate::Function`], and that's exactly what this
//! module does.
//!
//! As syntax trees are values and don't know their place of origin/identity,
//! this module also requires [`InFile`] wrappers to understand which specific
//! real or macro-expanded file the tree comes from.
//!
//! The actual algorithm to resolve syntax to def is curious in two aspects:
//!
//!     * It is recursive
//!     * It uses the inverse algorithm (what is the syntax for this def?)
//!
//! Specifically, the algorithm goes like this:
//!
//!     1. Find the syntactic container for the syntax. For example, field's
//!        container is the struct, and structs container is a module.
//!     2. Recursively get the def corresponding to container.
//!     3. Ask the container def for all child defs. These child defs contain
//!        the answer and answer's siblings.
//!     4. For each child def, ask for it's source.
//!     5. The child def whose source is the syntax node we've started with
//!        is the answer.
//!
//! It's interesting that both Roslyn and Kotlin contain very similar code
//! shape.
//!
//! Let's take a look at Roslyn:
//!
//!   <https://github.com/dotnet/roslyn/blob/36a0c338d6621cc5fe34b79d414074a95a6a489c/src/Compilers/CSharp/Portable/Compilation/SyntaxTreeSemanticModel.cs#L1403-L1429>
//!   <https://sourceroslyn.io/#Microsoft.CodeAnalysis.CSharp/Compilation/SyntaxTreeSemanticModel.cs,1403>
//!
//! The `GetDeclaredType` takes `Syntax` as input, and returns `Symbol` as
//! output. First, it retrieves a `Symbol` for parent `Syntax`:
//!
//! * <https://sourceroslyn.io/#Microsoft.CodeAnalysis.CSharp/Compilation/SyntaxTreeSemanticModel.cs,1423>
//!
//! Then, it iterates parent symbol's children, looking for one which has the
//! same text span as the original node:
//!
//!   <https://sourceroslyn.io/#Microsoft.CodeAnalysis.CSharp/Compilation/SyntaxTreeSemanticModel.cs,1786>
//!
//! Now, let's look at Kotlin:
//!
//!   <https://github.com/JetBrains/kotlin/blob/a288b8b00e4754a1872b164999c6d3f3b8c8994a/idea/idea-frontend-fir/idea-fir-low-level-api/src/org/jetbrains/kotlin/idea/fir/low/level/api/FirModuleResolveStateImpl.kt#L93-L125>
//!
//! This function starts with a syntax node (`KtExpression` is syntax, like all
//! `Kt` nodes), and returns a def. It uses
//! `getNonLocalContainingOrThisDeclaration` to get syntactic container for a
//! current node. Then, `findSourceNonLocalFirDeclaration` gets `Fir` for this
//! parent. Finally, `findElementIn` function traverses `Fir` children to find
//! one with the same source we originally started with.
//!
//! One question is left though -- where does the recursion stops? This happens
//! when we get to the file syntax node, which doesn't have a syntactic parent.
//! In that case, we loop through all the crates that might contain this file
//! and look for a module whose source is the given file.
//!
//! Note that the logic in this module is somewhat fundamentally imprecise --
//! due to conditional compilation and `#[path]` attributes, there's no
//! injective mapping from syntax nodes to defs. This is not an edge case --
//! more or less every item in a `lib.rs` is a part of two distinct crates: a
//! library with `--cfg test` and a library without.
//!
//! At the moment, we don't really handle this well and return the first answer
//! that works. Ideally, we should first let the caller to pick a specific
//! active crate for a given position, and then provide an API to resolve all
//! syntax nodes against this specific crate.

use base_db::FileId;
use hir_def::{
    attr::AttrId,
    child_by_source::ChildBySource,
    dyn_map::DynMap,
    expr::{LabelId, PatId},
    keys::{self, Key},
    AdtId, ConstId, ConstParamId, DefWithBodyId, EnumId, EnumVariantId, FieldId, FunctionId,
    GenericDefId, GenericParamId, ImplId, LifetimeParamId, ModuleId, StaticId, StructId, TraitId,
    TypeAliasId, TypeParamId, UnionId, VariantId,
};
use hir_expand::{name::AsName, AstId, HirFileId, MacroCallId, MacroDefId, MacroDefKind};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use stdx::impl_from;
use syntax::{
    ast::{self, HasName},
    match_ast, AstNode, SyntaxNode,
};

use crate::{db::HirDatabase, InFile};

pub(super) type SourceToDefCache = FxHashMap<(ChildContainer, HirFileId), DynMap>;

pub(super) struct SourceToDefCtx<'a, 'b> {
    pub(super) db: &'b dyn HirDatabase,
    pub(super) cache: &'a mut SourceToDefCache,
}

impl SourceToDefCtx<'_, '_> {
    pub(super) fn file_to_def(&mut self, file: FileId) -> SmallVec<[ModuleId; 1]> {
        let _p = profile::span("SourceBinder::to_module_def");
        let mut mods = SmallVec::new();
        for &crate_id in self.db.relevant_crates(file).iter() {
            // FIXME: inner items
            let crate_def_map = self.db.crate_def_map(crate_id);
            mods.extend(
                crate_def_map
                    .modules_for_file(file)
                    .map(|local_id| crate_def_map.module_id(local_id)),
            )
        }
        mods
    }

    pub(super) fn module_to_def(&mut self, src: InFile<ast::Module>) -> Option<ModuleId> {
        let _p = profile::span("module_to_def");
        let parent_declaration =
            src.syntax().ancestors_with_macros_skip_attr_item(self.db.upcast()).skip(1).find_map(
                |it| {
                    let m = ast::Module::cast(it.value.clone())?;
                    Some(it.with_value(m))
                },
            );

        let parent_module = match parent_declaration {
            Some(parent_declaration) => self.module_to_def(parent_declaration),
            None => {
                let file_id = src.file_id.original_file(self.db.upcast());
                self.file_to_def(file_id).get(0).copied()
            }
        }?;

        let child_name = src.value.name()?.as_name();
        let def_map = parent_module.def_map(self.db.upcast());
        let child_id = *def_map[parent_module.local_id].children.get(&child_name)?;
        Some(def_map.module_id(child_id))
    }

    pub(super) fn source_file_to_def(&mut self, src: InFile<ast::SourceFile>) -> Option<ModuleId> {
        let _p = profile::span("source_file_to_def");
        let file_id = src.file_id.original_file(self.db.upcast());
        self.file_to_def(file_id).get(0).copied()
    }

    pub(super) fn trait_to_def(&mut self, src: InFile<ast::Trait>) -> Option<TraitId> {
        self.to_def(src, keys::TRAIT)
    }
    pub(super) fn impl_to_def(&mut self, src: InFile<ast::Impl>) -> Option<ImplId> {
        self.to_def(src, keys::IMPL)
    }
    pub(super) fn fn_to_def(&mut self, src: InFile<ast::Fn>) -> Option<FunctionId> {
        self.to_def(src, keys::FUNCTION)
    }
    pub(super) fn struct_to_def(&mut self, src: InFile<ast::Struct>) -> Option<StructId> {
        self.to_def(src, keys::STRUCT)
    }
    pub(super) fn enum_to_def(&mut self, src: InFile<ast::Enum>) -> Option<EnumId> {
        self.to_def(src, keys::ENUM)
    }
    pub(super) fn union_to_def(&mut self, src: InFile<ast::Union>) -> Option<UnionId> {
        self.to_def(src, keys::UNION)
    }
    pub(super) fn static_to_def(&mut self, src: InFile<ast::Static>) -> Option<StaticId> {
        self.to_def(src, keys::STATIC)
    }
    pub(super) fn const_to_def(&mut self, src: InFile<ast::Const>) -> Option<ConstId> {
        self.to_def(src, keys::CONST)
    }
    pub(super) fn type_alias_to_def(&mut self, src: InFile<ast::TypeAlias>) -> Option<TypeAliasId> {
        self.to_def(src, keys::TYPE_ALIAS)
    }
    pub(super) fn record_field_to_def(&mut self, src: InFile<ast::RecordField>) -> Option<FieldId> {
        self.to_def(src, keys::RECORD_FIELD)
    }
    pub(super) fn tuple_field_to_def(&mut self, src: InFile<ast::TupleField>) -> Option<FieldId> {
        self.to_def(src, keys::TUPLE_FIELD)
    }
    pub(super) fn enum_variant_to_def(
        &mut self,
        src: InFile<ast::Variant>,
    ) -> Option<EnumVariantId> {
        self.to_def(src, keys::VARIANT)
    }
    pub(super) fn adt_to_def(
        &mut self,
        InFile { file_id, value }: InFile<ast::Adt>,
    ) -> Option<AdtId> {
        match value {
            ast::Adt::Enum(it) => self.enum_to_def(InFile::new(file_id, it)).map(AdtId::EnumId),
            ast::Adt::Struct(it) => {
                self.struct_to_def(InFile::new(file_id, it)).map(AdtId::StructId)
            }
            ast::Adt::Union(it) => self.union_to_def(InFile::new(file_id, it)).map(AdtId::UnionId),
        }
    }
    pub(super) fn bind_pat_to_def(
        &mut self,
        src: InFile<ast::IdentPat>,
    ) -> Option<(DefWithBodyId, PatId)> {
        let container = self.find_pat_or_label_container(src.syntax())?;
        let (_body, source_map) = self.db.body_with_source_map(container);
        let src = src.map(ast::Pat::from);
        let pat_id = source_map.node_pat(src.as_ref())?;
        Some((container, pat_id))
    }
    pub(super) fn self_param_to_def(
        &mut self,
        src: InFile<ast::SelfParam>,
    ) -> Option<(DefWithBodyId, PatId)> {
        let container = self.find_pat_or_label_container(src.syntax())?;
        let (_body, source_map) = self.db.body_with_source_map(container);
        let pat_id = source_map.node_self_param(src.as_ref())?;
        Some((container, pat_id))
    }
    pub(super) fn label_to_def(
        &mut self,
        src: InFile<ast::Label>,
    ) -> Option<(DefWithBodyId, LabelId)> {
        let container = self.find_pat_or_label_container(src.syntax())?;
        let (_body, source_map) = self.db.body_with_source_map(container);
        let label_id = source_map.node_label(src.as_ref())?;
        Some((container, label_id))
    }

    pub(super) fn item_to_macro_call(&mut self, src: InFile<ast::Item>) -> Option<MacroCallId> {
        let map = self.dyn_map(src.as_ref())?;
        map[keys::ATTR_MACRO_CALL].get(&src.value).copied()
    }

    pub(super) fn attr_to_derive_macro_call(
        &mut self,
        item: InFile<&ast::Adt>,
        src: InFile<ast::Attr>,
    ) -> Option<(AttrId, &[Option<MacroCallId>])> {
        let map = self.dyn_map(item)?;
        map[keys::DERIVE_MACRO_CALL].get(&src.value).map(|(id, ids)| (*id, &**ids))
    }

    fn to_def<Ast: AstNode + 'static, ID: Copy + 'static>(
        &mut self,
        src: InFile<Ast>,
        key: Key<Ast, ID>,
    ) -> Option<ID> {
        self.dyn_map(src.as_ref())?[key].get(&src.value).copied()
    }

    fn dyn_map<Ast: AstNode + 'static>(&mut self, src: InFile<&Ast>) -> Option<&DynMap> {
        let container = self.find_container(src.map(|it| it.syntax()))?;
        Some(self.cache_for(container, src.file_id))
    }

    fn cache_for(&mut self, container: ChildContainer, file_id: HirFileId) -> &DynMap {
        let db = self.db;
        self.cache
            .entry((container, file_id))
            .or_insert_with(|| container.child_by_source(db, file_id))
    }

    pub(super) fn type_param_to_def(&mut self, src: InFile<ast::TypeParam>) -> Option<TypeParamId> {
        let container: ChildContainer = self.find_generic_param_container(src.syntax())?.into();
        let dyn_map = self.cache_for(container, src.file_id);
        dyn_map[keys::TYPE_PARAM].get(&src.value).copied()
    }

    pub(super) fn lifetime_param_to_def(
        &mut self,
        src: InFile<ast::LifetimeParam>,
    ) -> Option<LifetimeParamId> {
        let container: ChildContainer = self.find_generic_param_container(src.syntax())?.into();
        let dyn_map = self.cache_for(container, src.file_id);
        dyn_map[keys::LIFETIME_PARAM].get(&src.value).copied()
    }

    pub(super) fn const_param_to_def(
        &mut self,
        src: InFile<ast::ConstParam>,
    ) -> Option<ConstParamId> {
        let container: ChildContainer = self.find_generic_param_container(src.syntax())?.into();
        let dyn_map = self.cache_for(container, src.file_id);
        dyn_map[keys::CONST_PARAM].get(&src.value).copied()
    }

    pub(super) fn generic_param_to_def(
        &mut self,
        InFile { file_id, value }: InFile<ast::GenericParam>,
    ) -> Option<GenericParamId> {
        match value {
            ast::GenericParam::ConstParam(it) => {
                self.const_param_to_def(InFile::new(file_id, it)).map(GenericParamId::ConstParamId)
            }
            ast::GenericParam::LifetimeParam(it) => self
                .lifetime_param_to_def(InFile::new(file_id, it))
                .map(GenericParamId::LifetimeParamId),
            ast::GenericParam::TypeParam(it) => {
                self.type_param_to_def(InFile::new(file_id, it)).map(GenericParamId::TypeParamId)
            }
        }
    }

    pub(super) fn macro_to_def(&mut self, src: InFile<ast::Macro>) -> Option<MacroDefId> {
        let makro = self.dyn_map(src.as_ref()).and_then(|it| it[keys::MACRO].get(&src.value));
        if let Some(&makro) = makro {
            return Some(makro);
        }

        // Not all macros are recorded in the dyn map, only the ones behaving like items, so fall back
        // for the non-item like definitions.
        let file_ast_id = self.db.ast_id_map(src.file_id).ast_id(&src.value);
        let ast_id = AstId::new(src.file_id, file_ast_id.upcast());
        let kind = MacroDefKind::Declarative(ast_id);
        let file_id = src.file_id.original_file(self.db.upcast());
        let krate = self.file_to_def(file_id).get(0).copied()?.krate();
        Some(MacroDefId { krate, kind, local_inner: false })
    }

    pub(super) fn find_container(&mut self, src: InFile<&SyntaxNode>) -> Option<ChildContainer> {
        for container in src.ancestors_with_macros_skip_attr_item(self.db.upcast()).skip(1) {
            if let Some(res) = self.container_to_def(container) {
                return Some(res);
            }
        }

        let def = self.file_to_def(src.file_id.original_file(self.db.upcast())).get(0).copied()?;
        Some(def.into())
    }

    fn container_to_def(&mut self, container: InFile<SyntaxNode>) -> Option<ChildContainer> {
        let cont = match_ast! {
            match (container.value) {
                ast::Module(it) => {
                    let def = self.module_to_def(container.with_value(it))?;
                    def.into()
                },
                ast::Trait(it) => {
                    let def = self.trait_to_def(container.with_value(it))?;
                    def.into()
                },
                ast::Impl(it) => {
                    let def = self.impl_to_def(container.with_value(it))?;
                    def.into()
                },
                ast::Fn(it) => {
                    let def = self.fn_to_def(container.with_value(it))?;
                    DefWithBodyId::from(def).into()
                },
                ast::Struct(it) => {
                    let def = self.struct_to_def(container.with_value(it))?;
                    VariantId::from(def).into()
                },
                ast::Enum(it) => {
                    let def = self.enum_to_def(container.with_value(it))?;
                    def.into()
                },
                ast::Union(it) => {
                    let def = self.union_to_def(container.with_value(it))?;
                    VariantId::from(def).into()
                },
                ast::Static(it) => {
                    let def = self.static_to_def(container.with_value(it))?;
                    DefWithBodyId::from(def).into()
                },
                ast::Const(it) => {
                    let def = self.const_to_def(container.with_value(it))?;
                    DefWithBodyId::from(def).into()
                },
                ast::TypeAlias(it) => {
                    let def = self.type_alias_to_def(container.with_value(it))?;
                    def.into()
                },
                ast::Variant(it) => {
                    let def = self.enum_variant_to_def(container.with_value(it))?;
                    VariantId::from(def).into()
                },
                _ => return None,
            }
        };
        Some(cont)
    }

    fn find_generic_param_container(&mut self, src: InFile<&SyntaxNode>) -> Option<GenericDefId> {
        for container in src.ancestors_with_macros_skip_attr_item(self.db.upcast()).skip(1) {
            let res: GenericDefId = match_ast! {
                match (container.value) {
                    ast::Fn(it) => self.fn_to_def(container.with_value(it))?.into(),
                    ast::Struct(it) => self.struct_to_def(container.with_value(it))?.into(),
                    ast::Enum(it) => self.enum_to_def(container.with_value(it))?.into(),
                    ast::Trait(it) => self.trait_to_def(container.with_value(it))?.into(),
                    ast::TypeAlias(it) => self.type_alias_to_def(container.with_value(it))?.into(),
                    ast::Impl(it) => self.impl_to_def(container.with_value(it))?.into(),
                    _ => continue,
                }
            };
            return Some(res);
        }
        None
    }

    fn find_pat_or_label_container(&mut self, src: InFile<&SyntaxNode>) -> Option<DefWithBodyId> {
        for container in src.ancestors_with_macros_skip_attr_item(self.db.upcast()).skip(1) {
            let res: DefWithBodyId = match_ast! {
                match (container.value) {
                    ast::Const(it) => self.const_to_def(container.with_value(it))?.into(),
                    ast::Static(it) => self.static_to_def(container.with_value(it))?.into(),
                    ast::Fn(it) => self.fn_to_def(container.with_value(it))?.into(),
                    _ => continue,
                }
            };
            return Some(res);
        }
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ChildContainer {
    DefWithBodyId(DefWithBodyId),
    ModuleId(ModuleId),
    TraitId(TraitId),
    ImplId(ImplId),
    EnumId(EnumId),
    VariantId(VariantId),
    TypeAliasId(TypeAliasId),
    /// XXX: this might be the same def as, for example an `EnumId`. However,
    /// here the children are generic parameters, and not, eg enum variants.
    GenericDefId(GenericDefId),
}
impl_from! {
    DefWithBodyId,
    ModuleId,
    TraitId,
    ImplId,
    EnumId,
    VariantId,
    TypeAliasId,
    GenericDefId
    for ChildContainer
}

impl ChildContainer {
    fn child_by_source(self, db: &dyn HirDatabase, file_id: HirFileId) -> DynMap {
        let db = db.upcast();
        match self {
            ChildContainer::DefWithBodyId(it) => it.child_by_source(db, file_id),
            ChildContainer::ModuleId(it) => it.child_by_source(db, file_id),
            ChildContainer::TraitId(it) => it.child_by_source(db, file_id),
            ChildContainer::ImplId(it) => it.child_by_source(db, file_id),
            ChildContainer::EnumId(it) => it.child_by_source(db, file_id),
            ChildContainer::VariantId(it) => it.child_by_source(db, file_id),
            ChildContainer::TypeAliasId(_) => DynMap::default(),
            ChildContainer::GenericDefId(it) => it.child_by_source(db, file_id),
        }
    }
}
