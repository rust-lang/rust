//! Maps *syntax* of various definitions to their semantic ids.
//!
//! This is a very interesting module, and, in some sense, can be considered the
//! heart of the IDE parts of rust-analyzer.
//!
//! This module solves the following problem:
//!
//! > Given a piece of syntax, find the corresponding semantic definition (def).
//!
//! This problem is a part of more-or-less every IDE feature implemented. Every
//! IDE functionality (like goto to definition), conceptually starts with a
//! specific cursor position in a file. Starting with this text offset, we first
//! figure out what syntactic construct are we at: is this a pattern, an
//! expression, an item definition.
//!
//! Knowing only the syntax gives us relatively little info. For example,
//! looking at the syntax of the function we can realize that it is a part of an
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
//! * It is recursive
//! * It uses the inverse algorithm (what is the syntax for this def?)
//!
//! Specifically, the algorithm goes like this:
//!
//! 1. Find the syntactic container for the syntax. For example, field's
//!    container is the struct, and structs container is a module.
//! 2. Recursively get the def corresponding to container.
//! 3. Ask the container def for all child defs. These child defs contain
//!    the answer and answer's siblings.
//! 4. For each child def, ask for it's source.
//! 5. The child def whose source is the syntax node we've started with
//!    is the answer.
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

use either::Either;
use hir_def::{
    AdtId, BlockId, ConstId, ConstParamId, DefWithBodyId, EnumId, EnumVariantId, ExternBlockId,
    ExternCrateId, FieldId, FunctionId, GenericDefId, GenericParamId, ImplId, LifetimeParamId,
    Lookup, MacroId, ModuleId, StaticId, StructId, TraitAliasId, TraitId, TypeAliasId, TypeParamId,
    UnionId, UseId, VariantId,
    dyn_map::{
        DynMap,
        keys::{self, Key},
    },
    hir::{BindingId, Expr, LabelId},
    nameres::{block_def_map, crate_def_map},
};
use hir_expand::{
    EditionedFileId, ExpansionInfo, HirFileId, InMacroFile, MacroCallId, attrs::AttrId,
    name::AsName,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use span::FileId;
use stdx::impl_from;
use syntax::{
    AstNode, AstPtr, SyntaxNode,
    ast::{self, HasName},
};
use tt::TextRange;

use crate::{InFile, InlineAsmOperand, db::HirDatabase, semantics::child_by_source::ChildBySource};

#[derive(Default)]
pub(super) struct SourceToDefCache {
    pub(super) dynmap_cache: FxHashMap<(ChildContainer, HirFileId), DynMap>,
    expansion_info_cache: FxHashMap<MacroCallId, ExpansionInfo>,
    pub(super) file_to_def_cache: FxHashMap<FileId, SmallVec<[ModuleId; 1]>>,
    pub(super) included_file_cache: FxHashMap<EditionedFileId, Option<MacroCallId>>,
    /// Rootnode to HirFileId cache
    pub(super) root_to_file_cache: FxHashMap<SyntaxNode, HirFileId>,
}

impl SourceToDefCache {
    pub(super) fn cache(
        root_to_file_cache: &mut FxHashMap<SyntaxNode, HirFileId>,
        root_node: SyntaxNode,
        file_id: HirFileId,
    ) {
        assert!(root_node.parent().is_none());
        let prev = root_to_file_cache.insert(root_node, file_id);
        assert!(prev.is_none() || prev == Some(file_id));
    }

    pub(super) fn get_or_insert_include_for(
        &mut self,
        db: &dyn HirDatabase,
        file: EditionedFileId,
    ) -> Option<MacroCallId> {
        if let Some(&m) = self.included_file_cache.get(&file) {
            return m;
        }
        self.included_file_cache.insert(file, None);
        for &crate_id in db.relevant_crates(file.file_id(db)).iter() {
            db.include_macro_invoc(crate_id).iter().for_each(|&(macro_call_id, file_id)| {
                self.included_file_cache.insert(file_id, Some(macro_call_id));
            });
        }
        self.included_file_cache.get(&file).copied().flatten()
    }

    pub(super) fn get_or_insert_expansion(
        &mut self,
        db: &dyn HirDatabase,
        macro_file: MacroCallId,
    ) -> &ExpansionInfo {
        self.expansion_info_cache.entry(macro_file).or_insert_with(|| {
            let exp_info = macro_file.expansion_info(db);

            let InMacroFile { file_id, value } = exp_info.expanded();
            Self::cache(&mut self.root_to_file_cache, value, file_id.into());

            exp_info
        })
    }
}

pub(super) struct SourceToDefCtx<'db, 'cache> {
    pub(super) db: &'db dyn HirDatabase,
    pub(super) cache: &'cache mut SourceToDefCache,
}

impl SourceToDefCtx<'_, '_> {
    pub(super) fn file_to_def(&mut self, file: FileId) -> &SmallVec<[ModuleId; 1]> {
        let _p = tracing::info_span!("SourceToDefCtx::file_to_def").entered();
        self.cache.file_to_def_cache.entry(file).or_insert_with(|| {
            let mut mods = SmallVec::new();

            for &crate_id in self.db.relevant_crates(file).iter() {
                // Note: `mod` declarations in block modules cannot be supported here
                let crate_def_map = crate_def_map(self.db, crate_id);
                let n_mods = mods.len();
                let modules = |file| {
                    crate_def_map
                        .modules_for_file(self.db, file)
                        .map(|local_id| crate_def_map.module_id(local_id))
                };
                mods.extend(modules(file));
                if mods.len() == n_mods {
                    mods.extend(
                        self.db
                            .include_macro_invoc(crate_id)
                            .iter()
                            .filter(|&&(_, file_id)| file_id.file_id(self.db) == file)
                            .flat_map(|&(macro_call_id, file_id)| {
                                self.cache.included_file_cache.insert(file_id, Some(macro_call_id));
                                modules(
                                    macro_call_id
                                        .lookup(self.db)
                                        .kind
                                        .file_id()
                                        .original_file(self.db)
                                        .file_id(self.db),
                                )
                            }),
                    );
                }
            }
            if mods.is_empty() {
                // FIXME: detached file
            }
            mods
        })
    }

    pub(super) fn module_to_def(&mut self, src: InFile<&ast::Module>) -> Option<ModuleId> {
        let _p = tracing::info_span!("module_to_def").entered();
        let parent_declaration = self
            .parent_ancestors_with_macros(src.syntax_ref(), |_, ancestor, _| {
                ancestor.map(Either::<ast::Module, ast::BlockExpr>::cast).transpose()
            })
            .map(|it| it.transpose());

        let parent_module = match parent_declaration {
            Some(Either::Right(parent_block)) => self
                .block_to_def(parent_block.as_ref())
                .map(|block| block_def_map(self.db, block).root_module_id()),
            Some(Either::Left(parent_declaration)) => {
                self.module_to_def(parent_declaration.as_ref())
            }
            None => {
                let file_id = src.file_id.original_file(self.db);
                self.file_to_def(file_id.file_id(self.db)).first().copied()
            }
        }?;

        let child_name = src.value.name()?.as_name();
        let def_map = parent_module.def_map(self.db);
        let &child_id = def_map[parent_module.local_id].children.get(&child_name)?;
        Some(def_map.module_id(child_id))
    }

    pub(super) fn source_file_to_def(&mut self, src: InFile<&ast::SourceFile>) -> Option<ModuleId> {
        let _p = tracing::info_span!("source_file_to_def").entered();
        let file_id = src.file_id.original_file(self.db);
        self.file_to_def(file_id.file_id(self.db)).first().copied()
    }

    pub(super) fn trait_to_def(&mut self, src: InFile<&ast::Trait>) -> Option<TraitId> {
        self.to_def(src, keys::TRAIT)
    }
    pub(super) fn trait_alias_to_def(
        &mut self,
        src: InFile<&ast::TraitAlias>,
    ) -> Option<TraitAliasId> {
        self.to_def(src, keys::TRAIT_ALIAS)
    }
    pub(super) fn impl_to_def(&mut self, src: InFile<&ast::Impl>) -> Option<ImplId> {
        self.to_def(src, keys::IMPL)
    }
    pub(super) fn fn_to_def(&mut self, src: InFile<&ast::Fn>) -> Option<FunctionId> {
        self.to_def(src, keys::FUNCTION)
    }
    pub(super) fn struct_to_def(&mut self, src: InFile<&ast::Struct>) -> Option<StructId> {
        self.to_def(src, keys::STRUCT)
    }
    pub(super) fn enum_to_def(&mut self, src: InFile<&ast::Enum>) -> Option<EnumId> {
        self.to_def(src, keys::ENUM)
    }
    pub(super) fn union_to_def(&mut self, src: InFile<&ast::Union>) -> Option<UnionId> {
        self.to_def(src, keys::UNION)
    }
    pub(super) fn static_to_def(&mut self, src: InFile<&ast::Static>) -> Option<StaticId> {
        self.to_def(src, keys::STATIC)
    }
    pub(super) fn const_to_def(&mut self, src: InFile<&ast::Const>) -> Option<ConstId> {
        self.to_def(src, keys::CONST)
    }
    pub(super) fn type_alias_to_def(
        &mut self,
        src: InFile<&ast::TypeAlias>,
    ) -> Option<TypeAliasId> {
        self.to_def(src, keys::TYPE_ALIAS)
    }
    pub(super) fn record_field_to_def(
        &mut self,
        src: InFile<&ast::RecordField>,
    ) -> Option<FieldId> {
        self.to_def(src, keys::RECORD_FIELD)
    }
    pub(super) fn tuple_field_to_def(&mut self, src: InFile<&ast::TupleField>) -> Option<FieldId> {
        self.to_def(src, keys::TUPLE_FIELD)
    }
    pub(super) fn block_to_def(&mut self, src: InFile<&ast::BlockExpr>) -> Option<BlockId> {
        self.to_def(src, keys::BLOCK)
    }
    pub(super) fn enum_variant_to_def(
        &mut self,
        src: InFile<&ast::Variant>,
    ) -> Option<EnumVariantId> {
        self.to_def(src, keys::ENUM_VARIANT)
    }
    pub(super) fn extern_crate_to_def(
        &mut self,
        src: InFile<&ast::ExternCrate>,
    ) -> Option<ExternCrateId> {
        self.to_def(src, keys::EXTERN_CRATE)
    }
    pub(super) fn extern_block_to_def(
        &mut self,
        src: InFile<&ast::ExternBlock>,
    ) -> Option<ExternBlockId> {
        self.to_def(src, keys::EXTERN_BLOCK)
    }
    #[allow(dead_code)]
    pub(super) fn use_to_def(&mut self, src: InFile<&ast::Use>) -> Option<UseId> {
        self.to_def(src, keys::USE)
    }
    pub(super) fn adt_to_def(
        &mut self,
        InFile { file_id, value }: InFile<&ast::Adt>,
    ) -> Option<AdtId> {
        match value {
            ast::Adt::Enum(it) => self.enum_to_def(InFile::new(file_id, it)).map(AdtId::EnumId),
            ast::Adt::Struct(it) => {
                self.struct_to_def(InFile::new(file_id, it)).map(AdtId::StructId)
            }
            ast::Adt::Union(it) => self.union_to_def(InFile::new(file_id, it)).map(AdtId::UnionId),
        }
    }

    pub(super) fn asm_operand_to_def(
        &mut self,
        src: InFile<&ast::AsmOperandNamed>,
    ) -> Option<InlineAsmOperand> {
        let asm = src.value.syntax().parent().and_then(ast::AsmExpr::cast)?;
        let index = asm
            .asm_pieces()
            .filter_map(|it| match it {
                ast::AsmPiece::AsmOperandNamed(it) => Some(it),
                _ => None,
            })
            .position(|it| it == *src.value)?;
        let container = self.find_pat_or_label_container(src.syntax_ref())?;
        let source_map = self.db.body_with_source_map(container).1;
        let expr = source_map.node_expr(src.with_value(&ast::Expr::AsmExpr(asm)))?.as_expr()?;
        Some(InlineAsmOperand { owner: container, expr, index })
    }

    pub(super) fn bind_pat_to_def(
        &mut self,
        src: InFile<&ast::IdentPat>,
    ) -> Option<(DefWithBodyId, BindingId)> {
        let container = self.find_pat_or_label_container(src.syntax_ref())?;
        let (body, source_map) = self.db.body_with_source_map(container);
        let src = src.cloned().map(ast::Pat::from);
        let pat_id = source_map.node_pat(src.as_ref())?;
        // the pattern could resolve to a constant, verify that this is not the case
        if let crate::Pat::Bind { id, .. } = body[pat_id.as_pat()?] {
            Some((container, id))
        } else {
            None
        }
    }
    pub(super) fn self_param_to_def(
        &mut self,
        src: InFile<&ast::SelfParam>,
    ) -> Option<(DefWithBodyId, BindingId)> {
        let container = self.find_pat_or_label_container(src.syntax_ref())?;
        let body = self.db.body(container);
        Some((container, body.self_param?))
    }
    pub(super) fn label_to_def(
        &mut self,
        src: InFile<&ast::Label>,
    ) -> Option<(DefWithBodyId, LabelId)> {
        let container = self.find_pat_or_label_container(src.syntax_ref())?;
        let source_map = self.db.body_with_source_map(container).1;

        let label_id = source_map.node_label(src)?;
        Some((container, label_id))
    }

    pub(super) fn label_ref_to_def(
        &mut self,
        src: InFile<&ast::Lifetime>,
    ) -> Option<(DefWithBodyId, LabelId)> {
        let break_or_continue = ast::Expr::cast(src.value.syntax().parent()?)?;
        let container = self.find_pat_or_label_container(src.syntax_ref())?;
        let (body, source_map) = self.db.body_with_source_map(container);
        let break_or_continue =
            source_map.node_expr(src.with_value(&break_or_continue))?.as_expr()?;
        let (Expr::Break { label, .. } | Expr::Continue { label }) = body[break_or_continue] else {
            return None;
        };
        Some((container, label?))
    }

    pub(super) fn item_to_macro_call(&mut self, src: InFile<&ast::Item>) -> Option<MacroCallId> {
        let map = self.dyn_map(src)?;
        map[keys::ATTR_MACRO_CALL].get(&AstPtr::new(src.value)).copied()
    }

    pub(super) fn macro_call_to_macro_call(
        &mut self,
        src: InFile<&ast::MacroCall>,
    ) -> Option<MacroCallId> {
        let map = self.dyn_map(src)?;
        map[keys::MACRO_CALL].get(&AstPtr::new(src.value)).copied()
    }

    /// (AttrId, derive attribute call id, derive call ids)
    pub(super) fn attr_to_derive_macro_call(
        &mut self,
        item: InFile<&ast::Adt>,
        src: InFile<ast::Attr>,
    ) -> Option<(AttrId, MacroCallId, &[Option<MacroCallId>])> {
        let map = self.dyn_map(item)?;
        map[keys::DERIVE_MACRO_CALL]
            .get(&AstPtr::new(&src.value))
            .map(|&(attr_id, call_id, ref ids)| (attr_id, call_id, &**ids))
    }

    pub(super) fn has_derives(&mut self, adt: InFile<&ast::Adt>) -> bool {
        self.dyn_map(adt).as_ref().is_some_and(|map| !map[keys::DERIVE_MACRO_CALL].is_empty())
    }

    fn to_def<Ast: AstNode + 'static, ID: Copy + 'static>(
        &mut self,
        src: InFile<&Ast>,
        key: Key<Ast, ID>,
    ) -> Option<ID> {
        self.dyn_map(src)?[key].get(&AstPtr::new(src.value)).copied()
    }

    fn dyn_map<Ast: AstNode + 'static>(&mut self, src: InFile<&Ast>) -> Option<&DynMap> {
        let container = self.find_container(src.map(|it| it.syntax()))?;
        Some(self.cache_for(container, src.file_id))
    }

    fn cache_for(&mut self, container: ChildContainer, file_id: HirFileId) -> &DynMap {
        let db = self.db;
        self.cache
            .dynmap_cache
            .entry((container, file_id))
            .or_insert_with(|| container.child_by_source(db, file_id))
    }

    pub(super) fn type_param_to_def(
        &mut self,
        src: InFile<&ast::TypeParam>,
    ) -> Option<TypeParamId> {
        let container: ChildContainer = self.find_generic_param_container(src.syntax_ref())?.into();
        let dyn_map = self.cache_for(container, src.file_id);
        dyn_map[keys::TYPE_PARAM]
            .get(&AstPtr::new(src.value))
            .copied()
            .map(TypeParamId::from_unchecked)
    }

    pub(super) fn lifetime_param_to_def(
        &mut self,
        src: InFile<&ast::LifetimeParam>,
    ) -> Option<LifetimeParamId> {
        let container: ChildContainer = self.find_generic_param_container(src.syntax_ref())?.into();
        let dyn_map = self.cache_for(container, src.file_id);
        dyn_map[keys::LIFETIME_PARAM].get(&AstPtr::new(src.value)).copied()
    }

    pub(super) fn const_param_to_def(
        &mut self,
        src: InFile<&ast::ConstParam>,
    ) -> Option<ConstParamId> {
        let container: ChildContainer = self.find_generic_param_container(src.syntax_ref())?.into();
        let dyn_map = self.cache_for(container, src.file_id);
        dyn_map[keys::CONST_PARAM]
            .get(&AstPtr::new(src.value))
            .copied()
            .map(ConstParamId::from_unchecked)
    }

    pub(super) fn generic_param_to_def(
        &mut self,
        InFile { file_id, value }: InFile<&ast::GenericParam>,
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

    pub(super) fn macro_to_def(&mut self, src: InFile<&ast::Macro>) -> Option<MacroId> {
        self.dyn_map(src).and_then(|it| match src.value {
            ast::Macro::MacroRules(value) => {
                it[keys::MACRO_RULES].get(&AstPtr::new(value)).copied().map(MacroId::from)
            }
            ast::Macro::MacroDef(value) => {
                it[keys::MACRO2].get(&AstPtr::new(value)).copied().map(MacroId::from)
            }
        })
    }

    pub(super) fn proc_macro_to_def(&mut self, src: InFile<&ast::Fn>) -> Option<MacroId> {
        self.dyn_map(src).and_then(|it| {
            it[keys::PROC_MACRO].get(&AstPtr::new(src.value)).copied().map(MacroId::from)
        })
    }

    pub(super) fn find_container(&mut self, src: InFile<&SyntaxNode>) -> Option<ChildContainer> {
        let _p = tracing::info_span!("find_container").entered();
        let def = self.parent_ancestors_with_macros(src, |this, container, child| {
            this.container_to_def(container, child)
        });
        if let Some(def) = def {
            return Some(def);
        }

        let def = self
            .file_to_def(src.file_id.original_file(self.db).file_id(self.db))
            .first()
            .copied()?;
        Some(def.into())
    }

    fn find_generic_param_container(&mut self, src: InFile<&SyntaxNode>) -> Option<GenericDefId> {
        self.parent_ancestors_with_macros(src, |this, InFile { file_id, value }, _| {
            let item = ast::Item::cast(value)?;
            match &item {
                ast::Item::Fn(it) => this.fn_to_def(InFile::new(file_id, it)).map(Into::into),
                ast::Item::Struct(it) => {
                    this.struct_to_def(InFile::new(file_id, it)).map(Into::into)
                }
                ast::Item::Enum(it) => this.enum_to_def(InFile::new(file_id, it)).map(Into::into),
                ast::Item::Trait(it) => this.trait_to_def(InFile::new(file_id, it)).map(Into::into),
                ast::Item::TraitAlias(it) => {
                    this.trait_alias_to_def(InFile::new(file_id, it)).map(Into::into)
                }
                ast::Item::TypeAlias(it) => {
                    this.type_alias_to_def(InFile::new(file_id, it)).map(Into::into)
                }
                ast::Item::Impl(it) => this.impl_to_def(InFile::new(file_id, it)).map(Into::into),
                _ => None,
            }
        })
    }

    // FIXME: Remove this when we do inference in signatures
    fn find_pat_or_label_container(&mut self, src: InFile<&SyntaxNode>) -> Option<DefWithBodyId> {
        self.parent_ancestors_with_macros(src, |this, InFile { file_id, value }, _| {
            let item = match ast::Item::cast(value.clone()) {
                Some(it) => it,
                None => {
                    let variant = ast::Variant::cast(value)?;
                    return this
                        .enum_variant_to_def(InFile::new(file_id, &variant))
                        .map(Into::into);
                }
            };
            match &item {
                ast::Item::Fn(it) => this.fn_to_def(InFile::new(file_id, it)).map(Into::into),
                ast::Item::Const(it) => this.const_to_def(InFile::new(file_id, it)).map(Into::into),
                ast::Item::Static(it) => {
                    this.static_to_def(InFile::new(file_id, it)).map(Into::into)
                }
                _ => None,
            }
        })
    }

    /// Skips the attributed item that caused the macro invocation we are climbing up
    fn parent_ancestors_with_macros<T>(
        &mut self,
        node: InFile<&SyntaxNode>,
        mut cb: impl FnMut(
            &mut Self,
            /*parent: */ InFile<SyntaxNode>,
            /*child: */ &SyntaxNode,
        ) -> Option<T>,
    ) -> Option<T> {
        let parent = |this: &mut Self, node: InFile<&SyntaxNode>| match node.value.parent() {
            Some(parent) => Some(node.with_value(parent)),
            None => {
                let macro_file = node.file_id.macro_file()?;
                let expansion_info = this.cache.get_or_insert_expansion(this.db, macro_file);
                expansion_info.arg().map(|node| node?.parent()).transpose()
            }
        };
        let mut deepest_child_in_same_file = node.cloned();
        let mut node = node.cloned();
        while let Some(parent) = parent(self, node.as_ref()) {
            if parent.file_id != node.file_id {
                deepest_child_in_same_file = parent.clone();
            }
            if let Some(res) = cb(self, parent.clone(), &deepest_child_in_same_file.value) {
                return Some(res);
            }
            node = parent;
        }
        None
    }

    fn container_to_def(
        &mut self,
        container: InFile<SyntaxNode>,
        child: &SyntaxNode,
    ) -> Option<ChildContainer> {
        let cont = if let Some(item) = ast::Item::cast(container.value.clone()) {
            match &item {
                ast::Item::Module(it) => self.module_to_def(container.with_value(it))?.into(),
                ast::Item::Trait(it) => self.trait_to_def(container.with_value(it))?.into(),
                ast::Item::TraitAlias(it) => {
                    self.trait_alias_to_def(container.with_value(it))?.into()
                }
                ast::Item::Impl(it) => self.impl_to_def(container.with_value(it))?.into(),
                ast::Item::Enum(it) => self.enum_to_def(container.with_value(it))?.into(),
                ast::Item::TypeAlias(it) => {
                    self.type_alias_to_def(container.with_value(it))?.into()
                }
                ast::Item::Struct(it) => {
                    let def = self.struct_to_def(container.with_value(it))?;
                    let is_in_body = it.field_list().is_some_and(|it| {
                        it.syntax().text_range().contains(child.text_range().start())
                    });
                    if is_in_body {
                        VariantId::from(def).into()
                    } else {
                        ChildContainer::GenericDefId(def.into())
                    }
                }
                ast::Item::Union(it) => {
                    let def = self.union_to_def(container.with_value(it))?;
                    let is_in_body = it.record_field_list().is_some_and(|it| {
                        it.syntax().text_range().contains(child.text_range().start())
                    });
                    if is_in_body {
                        VariantId::from(def).into()
                    } else {
                        ChildContainer::GenericDefId(def.into())
                    }
                }
                ast::Item::Fn(it) => {
                    let def = self.fn_to_def(container.with_value(it))?;
                    let child_offset = child.text_range().start();
                    let is_in_body =
                        it.body().is_some_and(|it| it.syntax().text_range().contains(child_offset));
                    let in_param_pat = || {
                        it.param_list().is_some_and(|it| {
                            it.self_param()
                                .and_then(|it| {
                                    Some(TextRange::new(
                                        it.syntax().text_range().start(),
                                        it.name()?.syntax().text_range().end(),
                                    ))
                                })
                                .is_some_and(|r| r.contains_inclusive(child_offset))
                                || it
                                    .params()
                                    .filter_map(|it| it.pat())
                                    .any(|it| it.syntax().text_range().contains(child_offset))
                        })
                    };
                    if is_in_body || in_param_pat() {
                        DefWithBodyId::from(def).into()
                    } else {
                        ChildContainer::GenericDefId(def.into())
                    }
                }
                ast::Item::Static(it) => {
                    let def = self.static_to_def(container.with_value(it))?;
                    let is_in_body = it.body().is_some_and(|it| {
                        it.syntax().text_range().contains(child.text_range().start())
                    });
                    if is_in_body {
                        DefWithBodyId::from(def).into()
                    } else {
                        ChildContainer::GenericDefId(def.into())
                    }
                }
                ast::Item::Const(it) => {
                    let def = self.const_to_def(container.with_value(it))?;
                    let is_in_body = it.body().is_some_and(|it| {
                        it.syntax().text_range().contains(child.text_range().start())
                    });
                    if is_in_body {
                        DefWithBodyId::from(def).into()
                    } else {
                        ChildContainer::GenericDefId(def.into())
                    }
                }
                _ => return None,
            }
        } else if let Some(it) = ast::Variant::cast(container.value.clone()) {
            let def = self.enum_variant_to_def(InFile::new(container.file_id, &it))?;
            let is_in_body =
                it.eq_token().is_some_and(|it| it.text_range().end() < child.text_range().start());
            if is_in_body { DefWithBodyId::from(def).into() } else { VariantId::from(def).into() }
        } else {
            let it = match Either::<ast::Pat, ast::Name>::cast(container.value)? {
                Either::Left(it) => ast::Param::cast(it.syntax().parent()?)?.syntax().parent(),
                Either::Right(it) => ast::SelfParam::cast(it.syntax().parent()?)?.syntax().parent(),
            }
            .and_then(ast::ParamList::cast)?
            .syntax()
            .parent()
            .and_then(ast::Fn::cast)?;
            let def = self.fn_to_def(InFile::new(container.file_id, &it))?;
            DefWithBodyId::from(def).into()
        };
        Some(cont)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ChildContainer {
    DefWithBodyId(DefWithBodyId),
    ModuleId(ModuleId),
    TraitId(TraitId),
    TraitAliasId(TraitAliasId),
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
    TraitAliasId,
    ImplId,
    EnumId,
    VariantId,
    TypeAliasId,
    GenericDefId
    for ChildContainer
}

impl ChildContainer {
    fn child_by_source(self, db: &dyn HirDatabase, file_id: HirFileId) -> DynMap {
        let _p = tracing::info_span!("ChildContainer::child_by_source").entered();
        match self {
            ChildContainer::DefWithBodyId(it) => it.child_by_source(db, file_id),
            ChildContainer::ModuleId(it) => it.child_by_source(db, file_id),
            ChildContainer::TraitId(it) => it.child_by_source(db, file_id),
            ChildContainer::TraitAliasId(_) => DynMap::default(),
            ChildContainer::ImplId(it) => it.child_by_source(db, file_id),
            ChildContainer::EnumId(it) => it.child_by_source(db, file_id),
            ChildContainer::VariantId(it) => it.child_by_source(db, file_id),
            ChildContainer::TypeAliasId(_) => DynMap::default(),
            ChildContainer::GenericDefId(it) => it.child_by_source(db, file_id),
        }
    }
}
