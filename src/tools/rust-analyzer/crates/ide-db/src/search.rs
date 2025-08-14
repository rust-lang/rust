//! Implementation of find-usages functionality.
//!
//! It is based on the standard ide trick: first, we run a fast text search to
//! get a super-set of matches. Then, we confirm each match using precise
//! name resolution.

use std::mem;
use std::{cell::LazyCell, cmp::Reverse};

use base_db::{RootQueryDb, SourceDatabase};
use either::Either;
use hir::{
    Adt, AsAssocItem, DefWithBody, EditionedFileId, FileRange, FileRangeWrapper, HasAttrs,
    HasContainer, HasSource, InFile, InFileWrapper, InRealFile, InlineAsmOperand, ItemContainer,
    ModuleSource, PathResolution, Semantics, Visibility, sym,
};
use memchr::memmem::Finder;
use parser::SyntaxKind;
use rustc_hash::{FxHashMap, FxHashSet};
use salsa::Database;
use syntax::{
    AstNode, AstToken, SmolStr, SyntaxElement, SyntaxNode, TextRange, TextSize, ToSmolStr,
    ast::{self, HasName, Rename},
    match_ast,
};
use triomphe::Arc;

use crate::{
    RootDatabase,
    defs::{Definition, NameClass, NameRefClass},
    traits::{as_trait_assoc_def, convert_to_def_in_trait},
};

#[derive(Debug, Default, Clone)]
pub struct UsageSearchResult {
    pub references: FxHashMap<EditionedFileId, Vec<FileReference>>,
}

impl UsageSearchResult {
    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
    }

    pub fn len(&self) -> usize {
        self.references.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (EditionedFileId, &[FileReference])> + '_ {
        self.references.iter().map(|(&file_id, refs)| (file_id, &**refs))
    }

    pub fn file_ranges(&self) -> impl Iterator<Item = FileRange> + '_ {
        self.references.iter().flat_map(|(&file_id, refs)| {
            refs.iter().map(move |&FileReference { range, .. }| FileRange { file_id, range })
        })
    }
}

impl IntoIterator for UsageSearchResult {
    type Item = (EditionedFileId, Vec<FileReference>);
    type IntoIter = <FxHashMap<EditionedFileId, Vec<FileReference>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.references.into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct FileReference {
    /// The range of the reference in the original file
    pub range: TextRange,
    /// The node of the reference in the (macro-)file
    pub name: FileReferenceNode,
    pub category: ReferenceCategory,
}

#[derive(Debug, Clone)]
pub enum FileReferenceNode {
    Name(ast::Name),
    NameRef(ast::NameRef),
    Lifetime(ast::Lifetime),
    FormatStringEntry(ast::String, TextRange),
}

impl FileReferenceNode {
    pub fn text_range(&self) -> TextRange {
        match self {
            FileReferenceNode::Name(it) => it.syntax().text_range(),
            FileReferenceNode::NameRef(it) => it.syntax().text_range(),
            FileReferenceNode::Lifetime(it) => it.syntax().text_range(),
            FileReferenceNode::FormatStringEntry(_, range) => *range,
        }
    }
    pub fn syntax(&self) -> SyntaxElement {
        match self {
            FileReferenceNode::Name(it) => it.syntax().clone().into(),
            FileReferenceNode::NameRef(it) => it.syntax().clone().into(),
            FileReferenceNode::Lifetime(it) => it.syntax().clone().into(),
            FileReferenceNode::FormatStringEntry(it, _) => it.syntax().clone().into(),
        }
    }
    pub fn into_name_like(self) -> Option<ast::NameLike> {
        match self {
            FileReferenceNode::Name(it) => Some(ast::NameLike::Name(it)),
            FileReferenceNode::NameRef(it) => Some(ast::NameLike::NameRef(it)),
            FileReferenceNode::Lifetime(it) => Some(ast::NameLike::Lifetime(it)),
            FileReferenceNode::FormatStringEntry(_, _) => None,
        }
    }
    pub fn as_name_ref(&self) -> Option<&ast::NameRef> {
        match self {
            FileReferenceNode::NameRef(name_ref) => Some(name_ref),
            _ => None,
        }
    }
    pub fn as_lifetime(&self) -> Option<&ast::Lifetime> {
        match self {
            FileReferenceNode::Lifetime(lifetime) => Some(lifetime),
            _ => None,
        }
    }
    pub fn text(&self) -> syntax::TokenText<'_> {
        match self {
            FileReferenceNode::NameRef(name_ref) => name_ref.text(),
            FileReferenceNode::Name(name) => name.text(),
            FileReferenceNode::Lifetime(lifetime) => lifetime.text(),
            FileReferenceNode::FormatStringEntry(it, range) => {
                syntax::TokenText::borrowed(&it.text()[*range - it.syntax().text_range().start()])
            }
        }
    }
}

bitflags::bitflags! {
    #[derive(Copy, Clone, Default, PartialEq, Eq, Hash, Debug)]
    pub struct ReferenceCategory: u8 {
        // FIXME: Add this variant and delete the `retain_adt_literal_usages` function.
        // const CREATE = 1 << 0;
        const WRITE = 1 << 0;
        const READ = 1 << 1;
        const IMPORT = 1 << 2;
        const TEST = 1 << 3;
    }
}

/// Generally, `search_scope` returns files that might contain references for the element.
/// For `pub(crate)` things it's a crate, for `pub` things it's a crate and dependant crates.
/// In some cases, the location of the references is known to within a `TextRange`,
/// e.g. for things like local variables.
#[derive(Clone, Debug)]
pub struct SearchScope {
    entries: FxHashMap<EditionedFileId, Option<TextRange>>,
}

impl SearchScope {
    fn new(entries: FxHashMap<EditionedFileId, Option<TextRange>>) -> SearchScope {
        SearchScope { entries }
    }

    /// Build a search scope spanning the entire crate graph of files.
    fn crate_graph(db: &RootDatabase) -> SearchScope {
        let mut entries = FxHashMap::default();

        let all_crates = db.all_crates();
        for &krate in all_crates.iter() {
            let crate_data = krate.data(db);
            let source_root = db.file_source_root(crate_data.root_file_id).source_root_id(db);
            let source_root = db.source_root(source_root).source_root(db);
            entries.extend(
                source_root
                    .iter()
                    .map(|id| (EditionedFileId::new(db, id, crate_data.edition), None)),
            );
        }
        SearchScope { entries }
    }

    /// Build a search scope spanning all the reverse dependencies of the given crate.
    fn reverse_dependencies(db: &RootDatabase, of: hir::Crate) -> SearchScope {
        let mut entries = FxHashMap::default();
        for rev_dep in of.transitive_reverse_dependencies(db) {
            let root_file = rev_dep.root_file(db);

            let source_root = db.file_source_root(root_file).source_root_id(db);
            let source_root = db.source_root(source_root).source_root(db);
            entries.extend(
                source_root
                    .iter()
                    .map(|id| (EditionedFileId::new(db, id, rev_dep.edition(db)), None)),
            );
        }
        SearchScope { entries }
    }

    /// Build a search scope spanning the given crate.
    fn krate(db: &RootDatabase, of: hir::Crate) -> SearchScope {
        let root_file = of.root_file(db);

        let source_root_id = db.file_source_root(root_file).source_root_id(db);
        let source_root = db.source_root(source_root_id).source_root(db);
        SearchScope {
            entries: source_root
                .iter()
                .map(|id| (EditionedFileId::new(db, id, of.edition(db)), None))
                .collect(),
        }
    }

    /// Build a search scope spanning the given module and all its submodules.
    pub fn module_and_children(db: &RootDatabase, module: hir::Module) -> SearchScope {
        let mut entries = FxHashMap::default();

        let (file_id, range) = {
            let InFile { file_id, value } = module.definition_source_range(db);
            if let Some(InRealFile { file_id, value: call_source }) = file_id.original_call_node(db)
            {
                (file_id, Some(call_source.text_range()))
            } else {
                (file_id.original_file(db), Some(value))
            }
        };
        entries.entry(file_id).or_insert(range);

        let mut to_visit: Vec<_> = module.children(db).collect();
        while let Some(module) = to_visit.pop() {
            if let Some(file_id) = module.as_source_file_id(db) {
                entries.insert(file_id, None);
            }
            to_visit.extend(module.children(db));
        }
        SearchScope { entries }
    }

    /// Build an empty search scope.
    pub fn empty() -> SearchScope {
        SearchScope::new(FxHashMap::default())
    }

    /// Build a empty search scope spanning the given file.
    pub fn single_file(file: EditionedFileId) -> SearchScope {
        SearchScope::new(std::iter::once((file, None)).collect())
    }

    /// Build a empty search scope spanning the text range of the given file.
    pub fn file_range(range: FileRange) -> SearchScope {
        SearchScope::new(std::iter::once((range.file_id, Some(range.range))).collect())
    }

    /// Build a empty search scope spanning the given files.
    pub fn files(files: &[EditionedFileId]) -> SearchScope {
        SearchScope::new(files.iter().map(|f| (*f, None)).collect())
    }

    pub fn intersection(&self, other: &SearchScope) -> SearchScope {
        let (mut small, mut large) = (&self.entries, &other.entries);
        if small.len() > large.len() {
            mem::swap(&mut small, &mut large)
        }

        let intersect_ranges =
            |r1: Option<TextRange>, r2: Option<TextRange>| -> Option<Option<TextRange>> {
                match (r1, r2) {
                    (None, r) | (r, None) => Some(r),
                    (Some(r1), Some(r2)) => r1.intersect(r2).map(Some),
                }
            };
        let res = small
            .iter()
            .filter_map(|(&file_id, &r1)| {
                let &r2 = large.get(&file_id)?;
                let r = intersect_ranges(r1, r2)?;
                Some((file_id, r))
            })
            .collect();

        SearchScope::new(res)
    }
}

impl IntoIterator for SearchScope {
    type Item = (EditionedFileId, Option<TextRange>);
    type IntoIter = std::collections::hash_map::IntoIter<EditionedFileId, Option<TextRange>>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl Definition {
    fn search_scope(&self, db: &RootDatabase) -> SearchScope {
        let _p = tracing::info_span!("search_scope").entered();

        if let Definition::BuiltinType(_) = self {
            return SearchScope::crate_graph(db);
        }

        // def is crate root
        if let &Definition::Module(module) = self
            && module.is_crate_root()
        {
            return SearchScope::reverse_dependencies(db, module.krate());
        }

        let module = match self.module(db) {
            Some(it) => it,
            None => return SearchScope::empty(),
        };
        let InFile { file_id, value: module_source } = module.definition_source(db);
        let file_id = file_id.original_file(db);

        if let Definition::Local(var) = self {
            let def = match var.parent(db) {
                DefWithBody::Function(f) => f.source(db).map(|src| src.syntax().cloned()),
                DefWithBody::Const(c) => c.source(db).map(|src| src.syntax().cloned()),
                DefWithBody::Static(s) => s.source(db).map(|src| src.syntax().cloned()),
                DefWithBody::Variant(v) => v.source(db).map(|src| src.syntax().cloned()),
            };
            return match def {
                Some(def) => SearchScope::file_range(
                    def.as_ref().original_file_range_with_macro_call_input(db),
                ),
                None => SearchScope::single_file(file_id),
            };
        }

        if let Definition::InlineAsmOperand(op) = self {
            let def = match op.parent(db) {
                DefWithBody::Function(f) => f.source(db).map(|src| src.syntax().cloned()),
                DefWithBody::Const(c) => c.source(db).map(|src| src.syntax().cloned()),
                DefWithBody::Static(s) => s.source(db).map(|src| src.syntax().cloned()),
                DefWithBody::Variant(v) => v.source(db).map(|src| src.syntax().cloned()),
            };
            return match def {
                Some(def) => SearchScope::file_range(
                    def.as_ref().original_file_range_with_macro_call_input(db),
                ),
                None => SearchScope::single_file(file_id),
            };
        }

        if let Definition::SelfType(impl_) = self {
            return match impl_.source(db).map(|src| src.syntax().cloned()) {
                Some(def) => SearchScope::file_range(
                    def.as_ref().original_file_range_with_macro_call_input(db),
                ),
                None => SearchScope::single_file(file_id),
            };
        }

        if let Definition::GenericParam(hir::GenericParam::LifetimeParam(param)) = self {
            let def = match param.parent(db) {
                hir::GenericDef::Function(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::Adt(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::Trait(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::TraitAlias(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::TypeAlias(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::Impl(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::Const(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::Static(it) => it.source(db).map(|src| src.syntax().cloned()),
            };
            return match def {
                Some(def) => SearchScope::file_range(
                    def.as_ref().original_file_range_with_macro_call_input(db),
                ),
                None => SearchScope::single_file(file_id),
            };
        }

        if let Definition::Macro(macro_def) = self {
            return match macro_def.kind(db) {
                hir::MacroKind::Declarative => {
                    if macro_def.attrs(db).by_key(sym::macro_export).exists() {
                        SearchScope::reverse_dependencies(db, module.krate())
                    } else {
                        SearchScope::krate(db, module.krate())
                    }
                }
                hir::MacroKind::AttrBuiltIn
                | hir::MacroKind::DeriveBuiltIn
                | hir::MacroKind::DeclarativeBuiltIn => SearchScope::crate_graph(db),
                hir::MacroKind::Derive | hir::MacroKind::Attr | hir::MacroKind::ProcMacro => {
                    SearchScope::reverse_dependencies(db, module.krate())
                }
            };
        }

        if let Definition::DeriveHelper(_) = self {
            return SearchScope::reverse_dependencies(db, module.krate());
        }

        let vis = self.visibility(db);
        if let Some(Visibility::Public) = vis {
            return SearchScope::reverse_dependencies(db, module.krate());
        }
        if let Some(Visibility::Module(module, _)) = vis {
            return SearchScope::module_and_children(db, module.into());
        }

        let range = match module_source {
            ModuleSource::Module(m) => Some(m.syntax().text_range()),
            ModuleSource::BlockExpr(b) => Some(b.syntax().text_range()),
            ModuleSource::SourceFile(_) => None,
        };
        match range {
            Some(range) => SearchScope::file_range(FileRange { file_id, range }),
            None => SearchScope::single_file(file_id),
        }
    }

    pub fn usages<'a>(self, sema: &'a Semantics<'_, RootDatabase>) -> FindUsages<'a> {
        FindUsages {
            def: self,
            rename: None,
            assoc_item_container: self.as_assoc_item(sema.db).map(|a| a.container(sema.db)),
            sema,
            scope: None,
            include_self_kw_refs: None,
            search_self_mod: false,
        }
    }
}

#[derive(Clone)]
pub struct FindUsages<'a> {
    def: Definition,
    rename: Option<&'a Rename>,
    sema: &'a Semantics<'a, RootDatabase>,
    scope: Option<&'a SearchScope>,
    /// The container of our definition should it be an assoc item
    assoc_item_container: Option<hir::AssocItemContainer>,
    /// whether to search for the `Self` type of the definition
    include_self_kw_refs: Option<hir::Type<'a>>,
    /// whether to search for the `self` module
    search_self_mod: bool,
}

impl<'a> FindUsages<'a> {
    /// Enable searching for `Self` when the definition is a type or `self` for modules.
    pub fn include_self_refs(mut self) -> Self {
        self.include_self_kw_refs = def_to_ty(self.sema, &self.def);
        self.search_self_mod = true;
        self
    }

    /// Limit the search to a given [`SearchScope`].
    pub fn in_scope(self, scope: &'a SearchScope) -> Self {
        self.set_scope(Some(scope))
    }

    /// Limit the search to a given [`SearchScope`].
    pub fn set_scope(mut self, scope: Option<&'a SearchScope>) -> Self {
        assert!(self.scope.is_none());
        self.scope = scope;
        self
    }

    // FIXME: This is just a temporary fix for not handling import aliases like
    // `use Foo as Bar`. We need to support them in a proper way.
    // See issue #14079
    pub fn with_rename(mut self, rename: Option<&'a Rename>) -> Self {
        self.rename = rename;
        self
    }

    pub fn at_least_one(&self) -> bool {
        let mut found = false;
        self.search(&mut |_, _| {
            found = true;
            true
        });
        found
    }

    pub fn all(self) -> UsageSearchResult {
        let mut res = UsageSearchResult::default();
        self.search(&mut |file_id, reference| {
            res.references.entry(file_id).or_default().push(reference);
            false
        });
        res
    }

    fn scope_files<'b>(
        db: &'b RootDatabase,
        scope: &'b SearchScope,
    ) -> impl Iterator<Item = (Arc<str>, EditionedFileId, TextRange)> + 'b {
        scope.entries.iter().map(|(&file_id, &search_range)| {
            let text = db.file_text(file_id.file_id(db)).text(db);
            let search_range =
                search_range.unwrap_or_else(|| TextRange::up_to(TextSize::of(&**text)));

            (text.clone(), file_id, search_range)
        })
    }

    fn match_indices<'b>(
        text: &'b str,
        finder: &'b Finder<'b>,
        search_range: TextRange,
    ) -> impl Iterator<Item = TextSize> + 'b {
        finder.find_iter(text.as_bytes()).filter_map(move |idx| {
            let offset: TextSize = idx.try_into().unwrap();
            if !search_range.contains_inclusive(offset) {
                return None;
            }
            // If this is not a word boundary, that means this is only part of an identifier,
            // so it can't be what we're looking for.
            // This speeds up short identifiers significantly.
            if text[..idx]
                .chars()
                .next_back()
                .is_some_and(|ch| matches!(ch, 'A'..='Z' | 'a'..='z' | '_'))
                || text[idx + finder.needle().len()..]
                    .chars()
                    .next()
                    .is_some_and(|ch| matches!(ch, 'A'..='Z' | 'a'..='z' | '_' | '0'..='9'))
            {
                return None;
            }
            Some(offset)
        })
    }

    fn find_nodes<'b>(
        sema: &'b Semantics<'_, RootDatabase>,
        name: &str,
        file_id: EditionedFileId,
        node: &syntax::SyntaxNode,
        offset: TextSize,
    ) -> impl Iterator<Item = SyntaxNode> + 'b {
        node.token_at_offset(offset)
            .find(|it| {
                // `name` is stripped of raw ident prefix. See the comment on name retrieval below.
                it.text().trim_start_matches('\'').trim_start_matches("r#") == name
            })
            .into_iter()
            .flat_map(move |token| {
                if sema.is_inside_macro_call(InFile::new(file_id.into(), &token)) {
                    sema.descend_into_macros_exact(token)
                } else {
                    <_>::from([token])
                }
                .into_iter()
                .filter_map(|it| it.parent())
            })
    }

    /// Performs a special fast search for associated functions. This is mainly intended
    /// to speed up `new()` which can take a long time.
    ///
    /// The trick is instead of searching for `func_name` search for `TypeThatContainsContainerName::func_name`.
    /// We cannot search exactly that (not even in tokens), because `ContainerName` may be aliased.
    /// Instead, we perform a textual search for `ContainerName`. Then, we look for all cases where
    /// `ContainerName` may be aliased (that includes `use ContainerName as Xyz` and
    /// `type Xyz = ContainerName`). We collect a list of all possible aliases of `ContainerName`.
    /// The list can have false positives (because there may be multiple types named `ContainerName`),
    /// but it cannot have false negatives. Then, we look for `TypeThatContainsContainerNameOrAnyAlias::func_name`.
    /// Those that will be found are of high chance to be actual hits (of course, we will need to verify
    /// that).
    ///
    /// Returns true if completed the search.
    // FIXME: Extend this to other cases, such as associated types/consts/enum variants (note those can be `use`d).
    fn short_associated_function_fast_search(
        &self,
        sink: &mut dyn FnMut(EditionedFileId, FileReference) -> bool,
        search_scope: &SearchScope,
        name: &str,
    ) -> bool {
        if self.scope.is_some() {
            return false;
        }

        let _p = tracing::info_span!("short_associated_function_fast_search").entered();

        let container = (|| {
            let Definition::Function(function) = self.def else {
                return None;
            };
            if function.has_self_param(self.sema.db) {
                return None;
            }
            match function.container(self.sema.db) {
                // Only freestanding `impl`s qualify; methods from trait
                // can be called from within subtraits and bounds.
                ItemContainer::Impl(impl_) => {
                    let has_trait = impl_.trait_(self.sema.db).is_some();
                    if has_trait {
                        return None;
                    }
                    let adt = impl_.self_ty(self.sema.db).as_adt()?;
                    Some(adt)
                }
                _ => None,
            }
        })();
        let Some(container) = container else {
            return false;
        };

        fn has_any_name(node: &SyntaxNode, mut predicate: impl FnMut(&str) -> bool) -> bool {
            node.descendants().any(|node| {
                match_ast! {
                    match node {
                        ast::Name(it) => predicate(it.text().trim_start_matches("r#")),
                        ast::NameRef(it) => predicate(it.text().trim_start_matches("r#")),
                        _ => false
                    }
                }
            })
        }

        // This is a fixpoint algorithm with O(number of aliases), but most types have no or few aliases,
        // so this should stay fast.
        //
        /// Returns `(aliases, ranges_where_Self_can_refer_to_our_type)`.
        fn collect_possible_aliases(
            sema: &Semantics<'_, RootDatabase>,
            container: Adt,
        ) -> Option<(FxHashSet<SmolStr>, Vec<FileRangeWrapper<EditionedFileId>>)> {
            fn insert_type_alias(
                db: &RootDatabase,
                to_process: &mut Vec<(SmolStr, SearchScope)>,
                alias_name: &str,
                def: Definition,
            ) {
                let alias = alias_name.trim_start_matches("r#").to_smolstr();
                tracing::debug!("found alias: {alias}");
                to_process.push((alias, def.search_scope(db)));
            }

            let _p = tracing::info_span!("collect_possible_aliases").entered();

            let db = sema.db;
            let container_name = container.name(db).as_str().to_smolstr();
            let search_scope = Definition::from(container).search_scope(db);
            let mut seen = FxHashSet::default();
            let mut completed = FxHashSet::default();
            let mut to_process = vec![(container_name, search_scope)];
            let mut is_possibly_self = Vec::new();
            let mut total_files_searched = 0;

            while let Some((current_to_process, current_to_process_search_scope)) = to_process.pop()
            {
                let is_alias = |alias: &ast::TypeAlias| {
                    let def = sema.to_def(alias)?;
                    let ty = def.ty(db);
                    let is_alias = ty.as_adt()? == container;
                    is_alias.then_some(def)
                };

                let finder = Finder::new(current_to_process.as_bytes());
                for (file_text, file_id, search_range) in
                    FindUsages::scope_files(db, &current_to_process_search_scope)
                {
                    let tree = LazyCell::new(move || sema.parse(file_id).syntax().clone());

                    for offset in FindUsages::match_indices(&file_text, &finder, search_range) {
                        let usages = FindUsages::find_nodes(
                            sema,
                            &current_to_process,
                            file_id,
                            &tree,
                            offset,
                        )
                        .filter(|it| matches!(it.kind(), SyntaxKind::NAME | SyntaxKind::NAME_REF));
                        for usage in usages {
                            if let Some(alias) = usage.parent().and_then(|it| {
                                let path = ast::PathSegment::cast(it)?.parent_path();
                                let use_tree = ast::UseTree::cast(path.syntax().parent()?)?;
                                use_tree.rename()?.name()
                            }) {
                                if seen.insert(InFileWrapper::new(
                                    file_id,
                                    alias.syntax().text_range(),
                                )) {
                                    tracing::debug!("found alias: {alias}");
                                    cov_mark::hit!(container_use_rename);
                                    // FIXME: `use`s have no easy way to determine their search scope, but they are rare.
                                    to_process.push((
                                        alias.text().to_smolstr(),
                                        current_to_process_search_scope.clone(),
                                    ));
                                }
                            } else if let Some(alias) =
                                usage.ancestors().find_map(ast::TypeAlias::cast)
                                && let Some(name) = alias.name()
                                && seen
                                    .insert(InFileWrapper::new(file_id, name.syntax().text_range()))
                            {
                                if let Some(def) = is_alias(&alias) {
                                    cov_mark::hit!(container_type_alias);
                                    insert_type_alias(
                                        sema.db,
                                        &mut to_process,
                                        name.text().as_str(),
                                        def.into(),
                                    );
                                } else {
                                    cov_mark::hit!(same_name_different_def_type_alias);
                                }
                            }

                            // We need to account for `Self`. It can only refer to our type inside an impl.
                            let impl_ = 'impl_: {
                                for ancestor in usage.ancestors() {
                                    if let Some(parent) = ancestor.parent()
                                        && let Some(parent) = ast::Impl::cast(parent)
                                    {
                                        // Only if the GENERIC_PARAM_LIST is directly under impl, otherwise it may be in the self ty.
                                        if matches!(
                                            ancestor.kind(),
                                            SyntaxKind::ASSOC_ITEM_LIST
                                                | SyntaxKind::WHERE_CLAUSE
                                                | SyntaxKind::GENERIC_PARAM_LIST
                                        ) {
                                            break;
                                        }
                                        if parent
                                            .trait_()
                                            .is_some_and(|trait_| *trait_.syntax() == ancestor)
                                        {
                                            break;
                                        }

                                        // Otherwise, found an impl where its self ty may be our type.
                                        break 'impl_ Some(parent);
                                    }
                                }
                                None
                            };
                            (|| {
                                let impl_ = impl_?;
                                is_possibly_self.push(sema.original_range(impl_.syntax()));
                                let assoc_items = impl_.assoc_item_list()?;
                                let type_aliases = assoc_items
                                    .syntax()
                                    .descendants()
                                    .filter_map(ast::TypeAlias::cast);
                                for type_alias in type_aliases {
                                    let Some(ty) = type_alias.ty() else { continue };
                                    let Some(name) = type_alias.name() else { continue };
                                    let contains_self = ty
                                        .syntax()
                                        .descendants_with_tokens()
                                        .any(|node| node.kind() == SyntaxKind::SELF_TYPE_KW);
                                    if !contains_self {
                                        continue;
                                    }
                                    if seen.insert(InFileWrapper::new(
                                        file_id,
                                        name.syntax().text_range(),
                                    )) {
                                        if let Some(def) = is_alias(&type_alias) {
                                            cov_mark::hit!(self_type_alias);
                                            insert_type_alias(
                                                sema.db,
                                                &mut to_process,
                                                name.text().as_str(),
                                                def.into(),
                                            );
                                        } else {
                                            cov_mark::hit!(same_name_different_def_type_alias);
                                        }
                                    }
                                }
                                Some(())
                            })();
                        }
                    }
                }

                completed.insert(current_to_process);

                total_files_searched += current_to_process_search_scope.entries.len();
                // FIXME: Maybe this needs to be relative to the project size, or at least to the initial search scope?
                if total_files_searched > 20_000 && completed.len() > 100 {
                    // This case is extremely unlikely (even searching for `Vec::new()` on rust-analyzer does not enter
                    // here - it searches less than 10,000 files, and it does so in five seconds), but if we get here,
                    // we at a risk of entering an almost-infinite loop of growing the aliases list. So just stop and
                    // let normal search handle this case.
                    tracing::info!(aliases_count = %completed.len(), "too much aliases; leaving fast path");
                    return None;
                }
            }

            // Impls can contain each other, so we need to deduplicate their ranges.
            is_possibly_self.sort_unstable_by_key(|position| {
                (position.file_id, position.range.start(), Reverse(position.range.end()))
            });
            is_possibly_self.dedup_by(|pos2, pos1| {
                pos1.file_id == pos2.file_id
                    && pos1.range.start() <= pos2.range.start()
                    && pos1.range.end() >= pos2.range.end()
            });

            tracing::info!(aliases_count = %completed.len(), "aliases search completed");

            Some((completed, is_possibly_self))
        }

        fn search(
            this: &FindUsages<'_>,
            finder: &Finder<'_>,
            name: &str,
            files: impl Iterator<Item = (Arc<str>, EditionedFileId, TextRange)>,
            mut container_predicate: impl FnMut(
                &SyntaxNode,
                InFileWrapper<EditionedFileId, TextRange>,
            ) -> bool,
            sink: &mut dyn FnMut(EditionedFileId, FileReference) -> bool,
        ) {
            for (file_text, file_id, search_range) in files {
                let tree = LazyCell::new(move || this.sema.parse(file_id).syntax().clone());

                for offset in FindUsages::match_indices(&file_text, finder, search_range) {
                    let usages = FindUsages::find_nodes(this.sema, name, file_id, &tree, offset)
                        .filter_map(ast::NameRef::cast);
                    for usage in usages {
                        let found_usage = usage
                            .syntax()
                            .parent()
                            .and_then(ast::PathSegment::cast)
                            .map(|path_segment| {
                                container_predicate(
                                    path_segment.parent_path().syntax(),
                                    InFileWrapper::new(file_id, usage.syntax().text_range()),
                                )
                            })
                            .unwrap_or(false);
                        if found_usage {
                            this.found_name_ref(&usage, sink);
                        }
                    }
                }
            }
        }

        let Some((container_possible_aliases, is_possibly_self)) =
            collect_possible_aliases(self.sema, container)
        else {
            return false;
        };

        cov_mark::hit!(short_associated_function_fast_search);

        // FIXME: If Rust ever gains the ability to `use Struct::method` we'll also need to account for free
        // functions.
        let finder = Finder::new(name.as_bytes());
        // The search for `Self` may return duplicate results with `ContainerName`, so deduplicate them.
        let mut self_positions = FxHashSet::default();
        tracing::info_span!("Self_search").in_scope(|| {
            search(
                self,
                &finder,
                name,
                is_possibly_self.into_iter().map(|position| {
                    (position.file_text(self.sema.db).clone(), position.file_id, position.range)
                }),
                |path, name_position| {
                    let has_self = path
                        .descendants_with_tokens()
                        .any(|node| node.kind() == SyntaxKind::SELF_TYPE_KW);
                    if has_self {
                        self_positions.insert(name_position);
                    }
                    has_self
                },
                sink,
            )
        });
        tracing::info_span!("aliases_search").in_scope(|| {
            search(
                self,
                &finder,
                name,
                FindUsages::scope_files(self.sema.db, search_scope),
                |path, name_position| {
                    has_any_name(path, |name| container_possible_aliases.contains(name))
                        && !self_positions.contains(&name_position)
                },
                sink,
            )
        });

        true
    }

    pub fn search(&self, sink: &mut dyn FnMut(EditionedFileId, FileReference) -> bool) {
        let _p = tracing::info_span!("FindUsages:search").entered();
        let sema = self.sema;

        let search_scope = {
            // FIXME: Is the trait scope needed for trait impl assoc items?
            let base =
                as_trait_assoc_def(sema.db, self.def).unwrap_or(self.def).search_scope(sema.db);
            match &self.scope {
                None => base,
                Some(scope) => base.intersection(scope),
            }
        };

        let name = match (self.rename, self.def) {
            (Some(rename), _) => {
                if rename.underscore_token().is_some() {
                    None
                } else {
                    rename.name().map(|n| n.to_smolstr())
                }
            }
            // special case crate modules as these do not have a proper name
            (_, Definition::Module(module)) if module.is_crate_root() => {
                // FIXME: This assumes the crate name is always equal to its display name when it
                // really isn't
                // we should instead look at the dependency edge name and recursively search our way
                // up the ancestors
                module
                    .krate()
                    .display_name(self.sema.db)
                    .map(|crate_name| crate_name.crate_name().symbol().as_str().into())
            }
            _ => {
                let self_kw_refs = || {
                    self.include_self_kw_refs.as_ref().and_then(|ty| {
                        ty.as_adt()
                            .map(|adt| adt.name(self.sema.db))
                            .or_else(|| ty.as_builtin().map(|builtin| builtin.name()))
                    })
                };
                // We need to search without the `r#`, hence `as_str` access.
                // We strip `'` from lifetimes and labels as otherwise they may not match with raw-escaped ones,
                // e.g. if we search `'foo` we won't find `'r#foo`.
                self.def
                    .name(sema.db)
                    .or_else(self_kw_refs)
                    .map(|it| it.as_str().trim_start_matches('\'').to_smolstr())
            }
        };
        let name = match &name {
            Some(s) => s.as_str(),
            None => return,
        };

        // FIXME: This should probably depend on the number of the results (specifically, the number of false results).
        if name.len() <= 7 && self.short_associated_function_fast_search(sink, &search_scope, name)
        {
            return;
        }

        let finder = &Finder::new(name);
        let include_self_kw_refs =
            self.include_self_kw_refs.as_ref().map(|ty| (ty, Finder::new("Self")));
        for (text, file_id, search_range) in Self::scope_files(sema.db, &search_scope) {
            let tree = LazyCell::new(move || sema.parse(file_id).syntax().clone());

            // Search for occurrences of the items name
            for offset in Self::match_indices(&text, finder, search_range) {
                let ret = tree.token_at_offset(offset).any(|token| {
                    if let Some((range, _frange, string_token, Some(nameres))) =
                        sema.check_for_format_args_template(token.clone(), offset)
                    {
                        return self.found_format_args_ref(
                            file_id,
                            range,
                            string_token,
                            nameres,
                            sink,
                        );
                    }
                    false
                });
                if ret {
                    return;
                }

                for name in Self::find_nodes(sema, name, file_id, &tree, offset)
                    .filter_map(ast::NameLike::cast)
                {
                    if match name {
                        ast::NameLike::NameRef(name_ref) => self.found_name_ref(&name_ref, sink),
                        ast::NameLike::Name(name) => self.found_name(&name, sink),
                        ast::NameLike::Lifetime(lifetime) => self.found_lifetime(&lifetime, sink),
                    } {
                        return;
                    }
                }
            }
            // Search for occurrences of the `Self` referring to our type
            if let Some((self_ty, finder)) = &include_self_kw_refs {
                for offset in Self::match_indices(&text, finder, search_range) {
                    for name_ref in Self::find_nodes(sema, "Self", file_id, &tree, offset)
                        .filter_map(ast::NameRef::cast)
                    {
                        if self.found_self_ty_name_ref(self_ty, &name_ref, sink) {
                            return;
                        }
                    }
                }
            }
        }

        // Search for `super` and `crate` resolving to our module
        if let Definition::Module(module) = self.def {
            let scope =
                search_scope.intersection(&SearchScope::module_and_children(self.sema.db, module));

            let is_crate_root = module.is_crate_root().then(|| Finder::new("crate"));
            let finder = &Finder::new("super");

            for (text, file_id, search_range) in Self::scope_files(sema.db, &scope) {
                self.sema.db.unwind_if_revision_cancelled();

                let tree = LazyCell::new(move || sema.parse(file_id).syntax().clone());

                for offset in Self::match_indices(&text, finder, search_range) {
                    for name_ref in Self::find_nodes(sema, "super", file_id, &tree, offset)
                        .filter_map(ast::NameRef::cast)
                    {
                        if self.found_name_ref(&name_ref, sink) {
                            return;
                        }
                    }
                }
                if let Some(finder) = &is_crate_root {
                    for offset in Self::match_indices(&text, finder, search_range) {
                        for name_ref in Self::find_nodes(sema, "crate", file_id, &tree, offset)
                            .filter_map(ast::NameRef::cast)
                        {
                            if self.found_name_ref(&name_ref, sink) {
                                return;
                            }
                        }
                    }
                }
            }
        }

        // search for module `self` references in our module's definition source
        match self.def {
            Definition::Module(module) if self.search_self_mod => {
                let src = module.definition_source(sema.db);
                let file_id = src.file_id.original_file(sema.db);
                let (file_id, search_range) = match src.value {
                    ModuleSource::Module(m) => (file_id, Some(m.syntax().text_range())),
                    ModuleSource::BlockExpr(b) => (file_id, Some(b.syntax().text_range())),
                    ModuleSource::SourceFile(_) => (file_id, None),
                };

                let search_range = if let Some(&range) = search_scope.entries.get(&file_id) {
                    match (range, search_range) {
                        (None, range) | (range, None) => range,
                        (Some(range), Some(search_range)) => match range.intersect(search_range) {
                            Some(range) => Some(range),
                            None => return,
                        },
                    }
                } else {
                    return;
                };

                let file_text = sema.db.file_text(file_id.file_id(self.sema.db));
                let text = file_text.text(sema.db);
                let search_range =
                    search_range.unwrap_or_else(|| TextRange::up_to(TextSize::of(&**text)));

                let tree = LazyCell::new(|| sema.parse(file_id).syntax().clone());
                let finder = &Finder::new("self");

                for offset in Self::match_indices(text, finder, search_range) {
                    for name_ref in Self::find_nodes(sema, "self", file_id, &tree, offset)
                        .filter_map(ast::NameRef::cast)
                    {
                        if self.found_self_module_name_ref(&name_ref, sink) {
                            return;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn found_self_ty_name_ref(
        &self,
        self_ty: &hir::Type<'_>,
        name_ref: &ast::NameRef,
        sink: &mut dyn FnMut(EditionedFileId, FileReference) -> bool,
    ) -> bool {
        // See https://github.com/rust-lang/rust-analyzer/pull/15864/files/e0276dc5ddc38c65240edb408522bb869f15afb4#r1389848845
        let ty_eq = |ty: hir::Type<'_>| match (ty.as_adt(), self_ty.as_adt()) {
            (Some(ty), Some(self_ty)) => ty == self_ty,
            (None, None) => ty == *self_ty,
            _ => false,
        };

        match NameRefClass::classify(self.sema, name_ref) {
            Some(NameRefClass::Definition(Definition::SelfType(impl_), _))
                if ty_eq(impl_.self_ty(self.sema.db)) =>
            {
                let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                let reference = FileReference {
                    range,
                    name: FileReferenceNode::NameRef(name_ref.clone()),
                    category: ReferenceCategory::empty(),
                };
                sink(file_id, reference)
            }
            _ => false,
        }
    }

    fn found_self_module_name_ref(
        &self,
        name_ref: &ast::NameRef,
        sink: &mut dyn FnMut(EditionedFileId, FileReference) -> bool,
    ) -> bool {
        match NameRefClass::classify(self.sema, name_ref) {
            Some(NameRefClass::Definition(def @ Definition::Module(_), _)) if def == self.def => {
                let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                let category = if is_name_ref_in_import(name_ref) {
                    ReferenceCategory::IMPORT
                } else {
                    ReferenceCategory::empty()
                };
                let reference = FileReference {
                    range,
                    name: FileReferenceNode::NameRef(name_ref.clone()),
                    category,
                };
                sink(file_id, reference)
            }
            _ => false,
        }
    }

    fn found_format_args_ref(
        &self,
        file_id: EditionedFileId,
        range: TextRange,
        token: ast::String,
        res: Either<PathResolution, InlineAsmOperand>,
        sink: &mut dyn FnMut(EditionedFileId, FileReference) -> bool,
    ) -> bool {
        let def = res.either(Definition::from, Definition::from);
        if def == self.def {
            let reference = FileReference {
                range,
                name: FileReferenceNode::FormatStringEntry(token, range),
                category: ReferenceCategory::READ,
            };
            sink(file_id, reference)
        } else {
            false
        }
    }

    fn found_lifetime(
        &self,
        lifetime: &ast::Lifetime,
        sink: &mut dyn FnMut(EditionedFileId, FileReference) -> bool,
    ) -> bool {
        match NameRefClass::classify_lifetime(self.sema, lifetime) {
            Some(NameRefClass::Definition(def, _)) if def == self.def => {
                let FileRange { file_id, range } = self.sema.original_range(lifetime.syntax());
                let reference = FileReference {
                    range,
                    name: FileReferenceNode::Lifetime(lifetime.clone()),
                    category: ReferenceCategory::empty(),
                };
                sink(file_id, reference)
            }
            _ => false,
        }
    }

    fn found_name_ref(
        &self,
        name_ref: &ast::NameRef,
        sink: &mut dyn FnMut(EditionedFileId, FileReference) -> bool,
    ) -> bool {
        match NameRefClass::classify(self.sema, name_ref) {
            Some(NameRefClass::Definition(def, _))
                if self.def == def
                    // is our def a trait assoc item? then we want to find all assoc items from trait impls of our trait
                    || matches!(self.assoc_item_container, Some(hir::AssocItemContainer::Trait(_)))
                        && convert_to_def_in_trait(self.sema.db, def) == self.def =>
            {
                let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                let reference = FileReference {
                    range,
                    name: FileReferenceNode::NameRef(name_ref.clone()),
                    category: ReferenceCategory::new(self.sema, &def, name_ref),
                };
                sink(file_id, reference)
            }
            // FIXME: special case type aliases, we can't filter between impl and trait defs here as we lack the substitutions
            // so we always resolve all assoc type aliases to both their trait def and impl defs
            Some(NameRefClass::Definition(def, _))
                if self.assoc_item_container.is_some()
                    && matches!(self.def, Definition::TypeAlias(_))
                    && convert_to_def_in_trait(self.sema.db, def)
                        == convert_to_def_in_trait(self.sema.db, self.def) =>
            {
                let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                let reference = FileReference {
                    range,
                    name: FileReferenceNode::NameRef(name_ref.clone()),
                    category: ReferenceCategory::new(self.sema, &def, name_ref),
                };
                sink(file_id, reference)
            }
            Some(NameRefClass::Definition(def, _)) if self.include_self_kw_refs.is_some() => {
                if self.include_self_kw_refs == def_to_ty(self.sema, &def) {
                    let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                    let reference = FileReference {
                        range,
                        name: FileReferenceNode::NameRef(name_ref.clone()),
                        category: ReferenceCategory::new(self.sema, &def, name_ref),
                    };
                    sink(file_id, reference)
                } else {
                    false
                }
            }
            Some(NameRefClass::FieldShorthand {
                local_ref: local,
                field_ref: field,
                adt_subst: _,
            }) => {
                let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());

                let field = Definition::Field(field);
                let local = Definition::Local(local);
                let access = match self.def {
                    Definition::Field(_) if field == self.def => {
                        ReferenceCategory::new(self.sema, &field, name_ref)
                    }
                    Definition::Local(_) if local == self.def => {
                        ReferenceCategory::new(self.sema, &local, name_ref)
                    }
                    _ => return false,
                };
                let reference = FileReference {
                    range,
                    name: FileReferenceNode::NameRef(name_ref.clone()),
                    category: access,
                };
                sink(file_id, reference)
            }
            _ => false,
        }
    }

    fn found_name(
        &self,
        name: &ast::Name,
        sink: &mut dyn FnMut(EditionedFileId, FileReference) -> bool,
    ) -> bool {
        match NameClass::classify(self.sema, name) {
            Some(NameClass::PatFieldShorthand { local_def: _, field_ref, adt_subst: _ })
                if matches!(
                    self.def, Definition::Field(_) if Definition::Field(field_ref) == self.def
                ) =>
            {
                let FileRange { file_id, range } = self.sema.original_range(name.syntax());
                let reference = FileReference {
                    range,
                    name: FileReferenceNode::Name(name.clone()),
                    // FIXME: mutable patterns should have `Write` access
                    category: ReferenceCategory::READ,
                };
                sink(file_id, reference)
            }
            Some(NameClass::ConstReference(def)) if self.def == def => {
                let FileRange { file_id, range } = self.sema.original_range(name.syntax());
                let reference = FileReference {
                    range,
                    name: FileReferenceNode::Name(name.clone()),
                    category: ReferenceCategory::empty(),
                };
                sink(file_id, reference)
            }
            Some(NameClass::Definition(def)) if def != self.def => {
                match (&self.assoc_item_container, self.def) {
                    // for type aliases we always want to reference the trait def and all the trait impl counterparts
                    // FIXME: only until we can resolve them correctly, see FIXME above
                    (Some(_), Definition::TypeAlias(_))
                        if convert_to_def_in_trait(self.sema.db, def)
                            != convert_to_def_in_trait(self.sema.db, self.def) =>
                    {
                        return false;
                    }
                    (Some(_), Definition::TypeAlias(_)) => {}
                    // We looking at an assoc item of a trait definition, so reference all the
                    // corresponding assoc items belonging to this trait's trait implementations
                    (Some(hir::AssocItemContainer::Trait(_)), _)
                        if convert_to_def_in_trait(self.sema.db, def) == self.def => {}
                    _ => return false,
                }
                let FileRange { file_id, range } = self.sema.original_range(name.syntax());
                let reference = FileReference {
                    range,
                    name: FileReferenceNode::Name(name.clone()),
                    category: ReferenceCategory::empty(),
                };
                sink(file_id, reference)
            }
            _ => false,
        }
    }
}

fn def_to_ty<'db>(sema: &Semantics<'db, RootDatabase>, def: &Definition) -> Option<hir::Type<'db>> {
    match def {
        Definition::Adt(adt) => Some(adt.ty(sema.db)),
        Definition::TypeAlias(it) => Some(it.ty(sema.db)),
        Definition::BuiltinType(it) => Some(it.ty(sema.db)),
        Definition::SelfType(it) => Some(it.self_ty(sema.db)),
        _ => None,
    }
}

impl ReferenceCategory {
    fn new(
        sema: &Semantics<'_, RootDatabase>,
        def: &Definition,
        r: &ast::NameRef,
    ) -> ReferenceCategory {
        let mut result = ReferenceCategory::empty();
        if is_name_ref_in_test(sema, r) {
            result |= ReferenceCategory::TEST;
        }

        // Only Locals and Fields have accesses for now.
        if !matches!(def, Definition::Local(_) | Definition::Field(_)) {
            if is_name_ref_in_import(r) {
                result |= ReferenceCategory::IMPORT;
            }
            return result;
        }

        let mode = r.syntax().ancestors().find_map(|node| {
            match_ast! {
                match node {
                    ast::BinExpr(expr) => {
                        if matches!(expr.op_kind()?, ast::BinaryOp::Assignment { .. }) {
                            // If the variable or field ends on the LHS's end then it's a Write
                            // (covers fields and locals). FIXME: This is not terribly accurate.
                            if let Some(lhs) = expr.lhs()
                                && lhs.syntax().text_range().end() == r.syntax().text_range().end() {
                                    return Some(ReferenceCategory::WRITE)
                                }
                        }
                        Some(ReferenceCategory::READ)
                    },
                    _ => None,
                }
            }
        }).unwrap_or(ReferenceCategory::READ);

        result | mode
    }
}

fn is_name_ref_in_import(name_ref: &ast::NameRef) -> bool {
    name_ref
        .syntax()
        .parent()
        .and_then(ast::PathSegment::cast)
        .and_then(|it| it.parent_path().top_path().syntax().parent())
        .is_some_and(|it| it.kind() == SyntaxKind::USE_TREE)
}

fn is_name_ref_in_test(sema: &Semantics<'_, RootDatabase>, name_ref: &ast::NameRef) -> bool {
    name_ref.syntax().ancestors().any(|node| match ast::Fn::cast(node) {
        Some(it) => sema.to_def(&it).is_some_and(|func| func.is_test(sema.db)),
        None => false,
    })
}
