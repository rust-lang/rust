//! A higher level attributes based on TokenTree, with also some shortcuts.

use std::{borrow::Cow, convert::identity, hash::Hash, ops};

use base_db::Crate;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{
    HirFileId, InFile,
    attrs::{Attr, AttrId, RawAttrs, collect_attrs},
    span_map::SpanMapRef,
};
use intern::{Symbol, sym};
use la_arena::{ArenaMap, Idx, RawIdx};
use mbe::DelimiterKind;
use rustc_abi::ReprOptions;
use span::AstIdNode;
use syntax::{
    AstPtr,
    ast::{self, HasAttrs},
};
use triomphe::Arc;
use tt::iter::{TtElement, TtIter};

use crate::{
    AdtId, AstIdLoc, AttrDefId, GenericParamId, HasModule, LocalFieldId, Lookup, MacroId,
    VariantId,
    db::DefDatabase,
    item_tree::block_item_tree_query,
    lang_item::LangItem,
    nameres::{ModuleOrigin, ModuleSource},
    src::{HasChildSource, HasSource},
};

/// Desugared attributes of an item post `cfg_attr` expansion.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Attrs(RawAttrs);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttrsWithOwner {
    attrs: Attrs,
    owner: AttrDefId,
}

impl Attrs {
    pub fn new(
        db: &dyn DefDatabase,
        owner: &dyn ast::HasAttrs,
        span_map: SpanMapRef<'_>,
        cfg_options: &CfgOptions,
    ) -> Self {
        Attrs(RawAttrs::new_expanded(db, owner, span_map, cfg_options))
    }

    pub fn get(&self, id: AttrId) -> Option<&Attr> {
        (**self).iter().find(|attr| attr.id == id)
    }

    pub(crate) fn expand_cfg_attr(
        db: &dyn DefDatabase,
        krate: Crate,
        raw_attrs: RawAttrs,
    ) -> Attrs {
        Attrs(raw_attrs.expand_cfg_attr(db, krate))
    }

    pub(crate) fn is_cfg_enabled_for(
        db: &dyn DefDatabase,
        owner: &dyn ast::HasAttrs,
        span_map: SpanMapRef<'_>,
        cfg_options: &CfgOptions,
    ) -> Result<(), CfgExpr> {
        RawAttrs::attrs_iter_expanded::<false>(db, owner, span_map, cfg_options)
            .filter_map(|attr| attr.cfg())
            .find_map(|cfg| match cfg_options.check(&cfg).is_none_or(identity) {
                true => None,
                false => Some(cfg),
            })
            .map_or(Ok(()), Err)
    }
}

impl ops::Deref for Attrs {
    type Target = [Attr];

    fn deref(&self) -> &[Attr] {
        &self.0
    }
}

impl ops::Deref for AttrsWithOwner {
    type Target = Attrs;

    fn deref(&self) -> &Attrs {
        &self.attrs
    }
}

impl Attrs {
    pub const EMPTY: Self = Self(RawAttrs::EMPTY);

    pub(crate) fn fields_attrs_query(
        db: &dyn DefDatabase,
        v: VariantId,
    ) -> Arc<ArenaMap<LocalFieldId, Attrs>> {
        let _p = tracing::info_span!("fields_attrs_query").entered();
        let mut res = ArenaMap::default();
        let (fields, file_id, krate) = match v {
            VariantId::EnumVariantId(it) => {
                let loc = it.lookup(db);
                let krate = loc.parent.lookup(db).container.krate;
                let source = loc.source(db);
                (source.value.field_list(), source.file_id, krate)
            }
            VariantId::StructId(it) => {
                let loc = it.lookup(db);
                let krate = loc.container.krate;
                let source = loc.source(db);
                (source.value.field_list(), source.file_id, krate)
            }
            VariantId::UnionId(it) => {
                let loc = it.lookup(db);
                let krate = loc.container.krate;
                let source = loc.source(db);
                (
                    source.value.record_field_list().map(ast::FieldList::RecordFieldList),
                    source.file_id,
                    krate,
                )
            }
        };
        let Some(fields) = fields else {
            return Arc::new(res);
        };

        let cfg_options = krate.cfg_options(db);
        let span_map = db.span_map(file_id);

        match fields {
            ast::FieldList::RecordFieldList(fields) => {
                let mut idx = 0;
                for field in fields.fields() {
                    let attrs =
                        Attrs(RawAttrs::new_expanded(db, &field, span_map.as_ref(), cfg_options));
                    if attrs.is_cfg_enabled(cfg_options).is_ok() {
                        res.insert(Idx::from_raw(RawIdx::from(idx)), attrs);
                        idx += 1;
                    }
                }
            }
            ast::FieldList::TupleFieldList(fields) => {
                let mut idx = 0;
                for field in fields.fields() {
                    let attrs =
                        Attrs(RawAttrs::new_expanded(db, &field, span_map.as_ref(), cfg_options));
                    if attrs.is_cfg_enabled(cfg_options).is_ok() {
                        res.insert(Idx::from_raw(RawIdx::from(idx)), attrs);
                        idx += 1;
                    }
                }
            }
        }

        res.shrink_to_fit();
        Arc::new(res)
    }
}

impl Attrs {
    #[inline]
    pub fn by_key(&self, key: Symbol) -> AttrQuery<'_> {
        AttrQuery { attrs: self, key }
    }

    #[inline]
    pub fn rust_analyzer_tool(&self) -> impl Iterator<Item = &Attr> {
        self.iter()
            .filter(|&attr| attr.path.segments().first().is_some_and(|s| *s == sym::rust_analyzer))
    }

    #[inline]
    pub fn cfg(&self) -> Option<CfgExpr> {
        let mut cfgs = self.by_key(sym::cfg).tt_values().map(CfgExpr::parse);
        let first = cfgs.next()?;
        match cfgs.next() {
            Some(second) => {
                let cfgs = [first, second].into_iter().chain(cfgs);
                Some(CfgExpr::All(cfgs.collect()))
            }
            None => Some(first),
        }
    }

    #[inline]
    pub fn cfgs(&self) -> impl Iterator<Item = CfgExpr> + '_ {
        self.by_key(sym::cfg).tt_values().map(CfgExpr::parse)
    }

    #[inline]
    pub(crate) fn is_cfg_enabled(&self, cfg_options: &CfgOptions) -> Result<(), CfgExpr> {
        self.cfgs().try_for_each(|cfg| {
            if cfg_options.check(&cfg) != Some(false) { Ok(()) } else { Err(cfg) }
        })
    }

    #[inline]
    pub fn lang(&self) -> Option<&Symbol> {
        self.by_key(sym::lang).string_value()
    }

    #[inline]
    pub fn lang_item(&self) -> Option<LangItem> {
        self.by_key(sym::lang).string_value().and_then(LangItem::from_symbol)
    }

    #[inline]
    pub fn has_doc_hidden(&self) -> bool {
        self.by_key(sym::doc).tt_values().any(|tt| {
            tt.top_subtree().delimiter.kind == DelimiterKind::Parenthesis &&
                matches!(tt.token_trees().flat_tokens(), [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] if ident.sym == sym::hidden)
        })
    }

    #[inline]
    pub fn has_doc_notable_trait(&self) -> bool {
        self.by_key(sym::doc).tt_values().any(|tt| {
            tt.top_subtree().delimiter.kind == DelimiterKind::Parenthesis &&
                matches!(tt.token_trees().flat_tokens(), [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] if ident.sym == sym::notable_trait)
        })
    }

    #[inline]
    pub fn doc_exprs(&self) -> impl Iterator<Item = DocExpr> + '_ {
        self.by_key(sym::doc).tt_values().map(DocExpr::parse)
    }

    #[inline]
    pub fn doc_aliases(&self) -> impl Iterator<Item = Symbol> + '_ {
        self.doc_exprs().flat_map(|doc_expr| doc_expr.aliases().to_vec())
    }

    #[inline]
    pub fn export_name(&self) -> Option<&Symbol> {
        self.by_key(sym::export_name).string_value()
    }

    #[inline]
    pub fn is_proc_macro(&self) -> bool {
        self.by_key(sym::proc_macro).exists()
    }

    #[inline]
    pub fn is_proc_macro_attribute(&self) -> bool {
        self.by_key(sym::proc_macro_attribute).exists()
    }

    #[inline]
    pub fn is_proc_macro_derive(&self) -> bool {
        self.by_key(sym::proc_macro_derive).exists()
    }

    #[inline]
    pub fn is_test(&self) -> bool {
        self.iter().any(|it| {
            it.path()
                .segments()
                .iter()
                .rev()
                .zip([sym::core, sym::prelude, sym::v1, sym::test].iter().rev())
                .all(|it| it.0 == it.1)
        })
    }

    #[inline]
    pub fn is_ignore(&self) -> bool {
        self.by_key(sym::ignore).exists()
    }

    #[inline]
    pub fn is_bench(&self) -> bool {
        self.by_key(sym::bench).exists()
    }

    #[inline]
    pub fn is_unstable(&self) -> bool {
        self.by_key(sym::unstable).exists()
    }

    #[inline]
    pub fn rustc_legacy_const_generics(&self) -> Option<Box<Box<[u32]>>> {
        self.by_key(sym::rustc_legacy_const_generics)
            .tt_values()
            .next()
            .map(parse_rustc_legacy_const_generics)
            .filter(|it| !it.is_empty())
            .map(Box::new)
    }

    #[inline]
    pub fn repr(&self) -> Option<ReprOptions> {
        self.by_key(sym::repr).tt_values().filter_map(parse_repr_tt).fold(None, |acc, repr| {
            acc.map_or(Some(repr), |mut acc| {
                merge_repr(&mut acc, repr);
                Some(acc)
            })
        })
    }
}

fn parse_rustc_legacy_const_generics(tt: &crate::tt::TopSubtree) -> Box<[u32]> {
    let mut indices = Vec::new();
    let mut iter = tt.iter();
    while let (Some(first), second) = (iter.next(), iter.next()) {
        match first {
            TtElement::Leaf(tt::Leaf::Literal(lit)) => match lit.symbol.as_str().parse() {
                Ok(index) => indices.push(index),
                Err(_) => break,
            },
            _ => break,
        }

        if let Some(comma) = second {
            match comma {
                TtElement::Leaf(tt::Leaf::Punct(punct)) if punct.char == ',' => {}
                _ => break,
            }
        }
    }

    indices.into_boxed_slice()
}

fn merge_repr(this: &mut ReprOptions, other: ReprOptions) {
    let ReprOptions { int, align, pack, flags, field_shuffle_seed: _ } = this;
    flags.insert(other.flags);
    *align = (*align).max(other.align);
    *pack = match (*pack, other.pack) {
        (Some(pack), None) | (None, Some(pack)) => Some(pack),
        _ => (*pack).min(other.pack),
    };
    if other.int.is_some() {
        *int = other.int;
    }
}

fn parse_repr_tt(tt: &crate::tt::TopSubtree) -> Option<ReprOptions> {
    use crate::builtin_type::{BuiltinInt, BuiltinUint};
    use rustc_abi::{Align, Integer, IntegerType, ReprFlags, ReprOptions};

    match tt.top_subtree().delimiter {
        tt::Delimiter { kind: DelimiterKind::Parenthesis, .. } => {}
        _ => return None,
    }

    let mut acc = ReprOptions::default();
    let mut tts = tt.iter();
    while let Some(tt) = tts.next() {
        let TtElement::Leaf(tt::Leaf::Ident(ident)) = tt else {
            continue;
        };
        let repr = match &ident.sym {
            s if *s == sym::packed => {
                let pack = if let Some(TtElement::Subtree(_, mut tt_iter)) = tts.peek() {
                    tts.next();
                    if let Some(TtElement::Leaf(tt::Leaf::Literal(lit))) = tt_iter.next() {
                        lit.symbol.as_str().parse().unwrap_or_default()
                    } else {
                        0
                    }
                } else {
                    0
                };
                let pack = Some(Align::from_bytes(pack).unwrap_or(Align::ONE));
                ReprOptions { pack, ..Default::default() }
            }
            s if *s == sym::align => {
                let mut align = None;
                if let Some(TtElement::Subtree(_, mut tt_iter)) = tts.peek() {
                    tts.next();
                    if let Some(TtElement::Leaf(tt::Leaf::Literal(lit))) = tt_iter.next()
                        && let Ok(a) = lit.symbol.as_str().parse()
                    {
                        align = Align::from_bytes(a).ok();
                    }
                }
                ReprOptions { align, ..Default::default() }
            }
            s if *s == sym::C => ReprOptions { flags: ReprFlags::IS_C, ..Default::default() },
            s if *s == sym::transparent => {
                ReprOptions { flags: ReprFlags::IS_TRANSPARENT, ..Default::default() }
            }
            s if *s == sym::simd => ReprOptions { flags: ReprFlags::IS_SIMD, ..Default::default() },
            repr => {
                let mut int = None;
                if let Some(builtin) = BuiltinInt::from_suffix_sym(repr)
                    .map(Either::Left)
                    .or_else(|| BuiltinUint::from_suffix_sym(repr).map(Either::Right))
                {
                    int = Some(match builtin {
                        Either::Left(bi) => match bi {
                            BuiltinInt::Isize => IntegerType::Pointer(true),
                            BuiltinInt::I8 => IntegerType::Fixed(Integer::I8, true),
                            BuiltinInt::I16 => IntegerType::Fixed(Integer::I16, true),
                            BuiltinInt::I32 => IntegerType::Fixed(Integer::I32, true),
                            BuiltinInt::I64 => IntegerType::Fixed(Integer::I64, true),
                            BuiltinInt::I128 => IntegerType::Fixed(Integer::I128, true),
                        },
                        Either::Right(bu) => match bu {
                            BuiltinUint::Usize => IntegerType::Pointer(false),
                            BuiltinUint::U8 => IntegerType::Fixed(Integer::I8, false),
                            BuiltinUint::U16 => IntegerType::Fixed(Integer::I16, false),
                            BuiltinUint::U32 => IntegerType::Fixed(Integer::I32, false),
                            BuiltinUint::U64 => IntegerType::Fixed(Integer::I64, false),
                            BuiltinUint::U128 => IntegerType::Fixed(Integer::I128, false),
                        },
                    });
                }
                ReprOptions { int, ..Default::default() }
            }
        };
        merge_repr(&mut acc, repr);
    }

    Some(acc)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DocAtom {
    /// eg. `#[doc(hidden)]`
    Flag(Symbol),
    /// eg. `#[doc(alias = "it")]`
    ///
    /// Note that a key can have multiple values that are all considered "active" at the same time.
    /// For example, `#[doc(alias = "x")]` and `#[doc(alias = "y")]`.
    KeyValue { key: Symbol, value: Symbol },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DocExpr {
    Invalid,
    /// eg. `#[doc(hidden)]`, `#[doc(alias = "x")]`
    Atom(DocAtom),
    /// eg. `#[doc(alias("x", "y"))]`
    Alias(Vec<Symbol>),
}

impl From<DocAtom> for DocExpr {
    fn from(atom: DocAtom) -> Self {
        DocExpr::Atom(atom)
    }
}

impl DocExpr {
    fn parse<S: Copy>(tt: &tt::TopSubtree<S>) -> DocExpr {
        next_doc_expr(tt.iter()).unwrap_or(DocExpr::Invalid)
    }

    pub fn aliases(&self) -> &[Symbol] {
        match self {
            DocExpr::Atom(DocAtom::KeyValue { key, value }) if *key == sym::alias => {
                std::slice::from_ref(value)
            }
            DocExpr::Alias(aliases) => aliases,
            _ => &[],
        }
    }
}

fn next_doc_expr<S: Copy>(mut it: TtIter<'_, S>) -> Option<DocExpr> {
    let name = match it.next() {
        None => return None,
        Some(TtElement::Leaf(tt::Leaf::Ident(ident))) => ident.sym.clone(),
        Some(_) => return Some(DocExpr::Invalid),
    };

    // Peek
    let ret = match it.peek() {
        Some(TtElement::Leaf(tt::Leaf::Punct(punct))) if punct.char == '=' => {
            it.next();
            match it.next() {
                Some(TtElement::Leaf(tt::Leaf::Literal(tt::Literal {
                    symbol: text,
                    kind: tt::LitKind::Str,
                    ..
                }))) => DocAtom::KeyValue { key: name, value: text.clone() }.into(),
                _ => return Some(DocExpr::Invalid),
            }
        }
        Some(TtElement::Subtree(_, subtree_iter)) => {
            it.next();
            let subs = parse_comma_sep(subtree_iter);
            match &name {
                s if *s == sym::alias => DocExpr::Alias(subs),
                _ => DocExpr::Invalid,
            }
        }
        _ => DocAtom::Flag(name).into(),
    };
    Some(ret)
}

fn parse_comma_sep<S>(iter: TtIter<'_, S>) -> Vec<Symbol> {
    iter.filter_map(|tt| match tt {
        TtElement::Leaf(tt::Leaf::Literal(tt::Literal {
            kind: tt::LitKind::Str, symbol, ..
        })) => Some(symbol.clone()),
        _ => None,
    })
    .collect()
}

impl AttrsWithOwner {
    pub fn new(db: &dyn DefDatabase, owner: AttrDefId) -> Self {
        Self { attrs: db.attrs(owner), owner }
    }

    pub(crate) fn attrs_query(db: &dyn DefDatabase, def: AttrDefId) -> Attrs {
        let _p = tracing::info_span!("attrs_query").entered();
        // FIXME: this should use `Trace` to avoid duplication in `source_map` below
        match def {
            AttrDefId::ModuleId(module) => {
                let def_map = module.def_map(db);
                let mod_data = &def_map[module.local_id];

                let raw_attrs = match mod_data.origin {
                    ModuleOrigin::File { definition, declaration_tree_id, declaration, .. } => {
                        let decl_attrs = declaration_tree_id
                            .item_tree(db)
                            .raw_attrs(declaration.upcast())
                            .clone();
                        let tree = db.file_item_tree(definition.into());
                        let def_attrs = tree.top_level_raw_attrs().clone();
                        decl_attrs.merge(def_attrs)
                    }
                    ModuleOrigin::CrateRoot { definition } => {
                        let tree = db.file_item_tree(definition.into());
                        tree.top_level_raw_attrs().clone()
                    }
                    ModuleOrigin::Inline { definition_tree_id, definition } => {
                        definition_tree_id.item_tree(db).raw_attrs(definition.upcast()).clone()
                    }
                    ModuleOrigin::BlockExpr { id, .. } => {
                        let tree = block_item_tree_query(db, id);
                        tree.top_level_raw_attrs().clone()
                    }
                };
                Attrs::expand_cfg_attr(db, module.krate, raw_attrs)
            }
            AttrDefId::FieldId(it) => db.fields_attrs(it.parent)[it.local_id].clone(),
            AttrDefId::EnumVariantId(it) => attrs_from_ast_id_loc(db, it),
            AttrDefId::AdtId(it) => match it {
                AdtId::StructId(it) => attrs_from_ast_id_loc(db, it),
                AdtId::EnumId(it) => attrs_from_ast_id_loc(db, it),
                AdtId::UnionId(it) => attrs_from_ast_id_loc(db, it),
            },
            AttrDefId::TraitId(it) => attrs_from_ast_id_loc(db, it),
            AttrDefId::TraitAliasId(it) => attrs_from_ast_id_loc(db, it),
            AttrDefId::MacroId(it) => match it {
                MacroId::Macro2Id(it) => attrs_from_ast_id_loc(db, it),
                MacroId::MacroRulesId(it) => attrs_from_ast_id_loc(db, it),
                MacroId::ProcMacroId(it) => attrs_from_ast_id_loc(db, it),
            },
            AttrDefId::ImplId(it) => attrs_from_ast_id_loc(db, it),
            AttrDefId::ConstId(it) => attrs_from_ast_id_loc(db, it),
            AttrDefId::StaticId(it) => attrs_from_ast_id_loc(db, it),
            AttrDefId::FunctionId(it) => attrs_from_ast_id_loc(db, it),
            AttrDefId::TypeAliasId(it) => attrs_from_ast_id_loc(db, it),
            AttrDefId::GenericParamId(it) => match it {
                GenericParamId::ConstParamId(it) => {
                    let src = it.parent().child_source(db);
                    // FIXME: We should be never getting `None` here.
                    Attrs(match src.value.get(it.local_id()) {
                        Some(val) => RawAttrs::new_expanded(
                            db,
                            val,
                            db.span_map(src.file_id).as_ref(),
                            def.krate(db).cfg_options(db),
                        ),
                        None => RawAttrs::EMPTY,
                    })
                }
                GenericParamId::TypeParamId(it) => {
                    let src = it.parent().child_source(db);
                    // FIXME: We should be never getting `None` here.
                    Attrs(match src.value.get(it.local_id()) {
                        Some(val) => RawAttrs::new_expanded(
                            db,
                            val,
                            db.span_map(src.file_id).as_ref(),
                            def.krate(db).cfg_options(db),
                        ),
                        None => RawAttrs::EMPTY,
                    })
                }
                GenericParamId::LifetimeParamId(it) => {
                    let src = it.parent.child_source(db);
                    // FIXME: We should be never getting `None` here.
                    Attrs(match src.value.get(it.local_id) {
                        Some(val) => RawAttrs::new_expanded(
                            db,
                            val,
                            db.span_map(src.file_id).as_ref(),
                            def.krate(db).cfg_options(db),
                        ),
                        None => RawAttrs::EMPTY,
                    })
                }
            },
            AttrDefId::ExternBlockId(it) => attrs_from_ast_id_loc(db, it),
            AttrDefId::ExternCrateId(it) => attrs_from_ast_id_loc(db, it),
            AttrDefId::UseId(it) => attrs_from_ast_id_loc(db, it),
        }
    }

    pub fn source_map(&self, db: &dyn DefDatabase) -> AttrSourceMap {
        let owner = match self.owner {
            AttrDefId::ModuleId(module) => {
                // Modules can have 2 attribute owners (the `mod x;` item, and the module file itself).

                let def_map = module.def_map(db);
                let mod_data = &def_map[module.local_id];
                match mod_data.declaration_source(db) {
                    Some(it) => {
                        let mut map = AttrSourceMap::new(InFile::new(it.file_id, &it.value));
                        if let InFile { file_id, value: ModuleSource::SourceFile(file) } =
                            mod_data.definition_source(db)
                        {
                            map.append_module_inline_attrs(AttrSourceMap::new(InFile::new(
                                file_id, &file,
                            )));
                        }
                        return map;
                    }
                    None => {
                        let InFile { file_id, value } = mod_data.definition_source(db);
                        let attrs_owner = match &value {
                            ModuleSource::SourceFile(file) => file as &dyn ast::HasAttrs,
                            ModuleSource::Module(module) => module as &dyn ast::HasAttrs,
                            ModuleSource::BlockExpr(block) => block as &dyn ast::HasAttrs,
                        };
                        return AttrSourceMap::new(InFile::new(file_id, attrs_owner));
                    }
                }
            }
            AttrDefId::FieldId(id) => {
                let map = db.fields_attrs_source_map(id.parent);
                let file_id = id.parent.file_id(db);
                let root = db.parse_or_expand(file_id);
                let owner = ast::AnyHasAttrs::new(map[id.local_id].to_node(&root));
                InFile::new(file_id, owner)
            }
            AttrDefId::AdtId(adt) => match adt {
                AdtId::StructId(id) => any_has_attrs(db, id),
                AdtId::UnionId(id) => any_has_attrs(db, id),
                AdtId::EnumId(id) => any_has_attrs(db, id),
            },
            AttrDefId::FunctionId(id) => any_has_attrs(db, id),
            AttrDefId::EnumVariantId(id) => any_has_attrs(db, id),
            AttrDefId::StaticId(id) => any_has_attrs(db, id),
            AttrDefId::ConstId(id) => any_has_attrs(db, id),
            AttrDefId::TraitId(id) => any_has_attrs(db, id),
            AttrDefId::TraitAliasId(id) => any_has_attrs(db, id),
            AttrDefId::TypeAliasId(id) => any_has_attrs(db, id),
            AttrDefId::MacroId(id) => match id {
                MacroId::Macro2Id(id) => any_has_attrs(db, id),
                MacroId::MacroRulesId(id) => any_has_attrs(db, id),
                MacroId::ProcMacroId(id) => any_has_attrs(db, id),
            },
            AttrDefId::ImplId(id) => any_has_attrs(db, id),
            AttrDefId::GenericParamId(id) => match id {
                GenericParamId::ConstParamId(id) => id
                    .parent()
                    .child_source(db)
                    .map(|source| ast::AnyHasAttrs::new(source[id.local_id()].clone())),
                GenericParamId::TypeParamId(id) => id
                    .parent()
                    .child_source(db)
                    .map(|source| ast::AnyHasAttrs::new(source[id.local_id()].clone())),
                GenericParamId::LifetimeParamId(id) => id
                    .parent
                    .child_source(db)
                    .map(|source| ast::AnyHasAttrs::new(source[id.local_id].clone())),
            },
            AttrDefId::ExternBlockId(id) => any_has_attrs(db, id),
            AttrDefId::ExternCrateId(id) => any_has_attrs(db, id),
            AttrDefId::UseId(id) => any_has_attrs(db, id),
        };

        AttrSourceMap::new(owner.as_ref().map(|node| node as &dyn HasAttrs))
    }
}

#[derive(Debug)]
pub struct AttrSourceMap {
    source: Vec<Either<ast::Attr, ast::Comment>>,
    file_id: HirFileId,
    /// If this map is for a module, this will be the [`HirFileId`] of the module's definition site,
    /// while `file_id` will be the one of the module declaration site.
    /// The usize is the index into `source` from which point on the entries reside in the def site
    /// file.
    mod_def_site_file_id: Option<(HirFileId, usize)>,
}

impl AttrSourceMap {
    fn new(owner: InFile<&dyn ast::HasAttrs>) -> Self {
        Self {
            source: collect_attrs(owner.value).map(|(_, it)| it).collect(),
            file_id: owner.file_id,
            mod_def_site_file_id: None,
        }
    }

    /// Append a second source map to this one, this is required for modules, whose outline and inline
    /// attributes can reside in different files
    fn append_module_inline_attrs(&mut self, other: Self) {
        assert!(self.mod_def_site_file_id.is_none() && other.mod_def_site_file_id.is_none());
        let len = self.source.len();
        self.source.extend(other.source);
        if other.file_id != self.file_id {
            self.mod_def_site_file_id = Some((other.file_id, len));
        }
    }

    /// Maps the lowered `Attr` back to its original syntax node.
    ///
    /// `attr` must come from the `owner` used for AttrSourceMap
    ///
    /// Note that the returned syntax node might be a `#[cfg_attr]`, or a doc comment, instead of
    /// the attribute represented by `Attr`.
    pub fn source_of(&self, attr: &Attr) -> InFile<&Either<ast::Attr, ast::Comment>> {
        self.source_of_id(attr.id)
    }

    pub fn source_of_id(&self, id: AttrId) -> InFile<&Either<ast::Attr, ast::Comment>> {
        let ast_idx = id.ast_index();
        let file_id = match self.mod_def_site_file_id {
            Some((file_id, def_site_cut)) if def_site_cut <= ast_idx => file_id,
            _ => self.file_id,
        };

        self.source
            .get(ast_idx)
            .map(|it| InFile::new(file_id, it))
            .unwrap_or_else(|| panic!("cannot find attr at index {id:?}"))
    }
}

#[derive(Debug, Clone)]
pub struct AttrQuery<'attr> {
    attrs: &'attr Attrs,
    key: Symbol,
}

impl<'attr> AttrQuery<'attr> {
    #[inline]
    pub fn tt_values(self) -> impl Iterator<Item = &'attr crate::tt::TopSubtree> {
        self.attrs().filter_map(|attr| attr.token_tree_value())
    }

    #[inline]
    pub fn string_value(self) -> Option<&'attr Symbol> {
        self.attrs().find_map(|attr| attr.string_value())
    }

    #[inline]
    pub fn string_value_with_span(self) -> Option<(&'attr Symbol, span::Span)> {
        self.attrs().find_map(|attr| attr.string_value_with_span())
    }

    #[inline]
    pub fn string_value_unescape(self) -> Option<Cow<'attr, str>> {
        self.attrs().find_map(|attr| attr.string_value_unescape())
    }

    #[inline]
    pub fn exists(self) -> bool {
        self.attrs().next().is_some()
    }

    #[inline]
    pub fn attrs(self) -> impl Iterator<Item = &'attr Attr> + Clone {
        let key = self.key;
        self.attrs.iter().filter(move |attr| attr.path.as_ident().is_some_and(|s| *s == key))
    }

    /// Find string value for a specific key inside token tree
    ///
    /// ```ignore
    /// #[doc(html_root_url = "url")]
    ///       ^^^^^^^^^^^^^ key
    /// ```
    #[inline]
    pub fn find_string_value_in_tt(self, key: Symbol) -> Option<&'attr str> {
        self.tt_values().find_map(|tt| {
            let name = tt.iter()
                .skip_while(|tt| !matches!(tt, TtElement::Leaf(tt::Leaf::Ident(tt::Ident { sym, ..} )) if *sym == key))
                .nth(2);

            match name {
                Some(TtElement::Leaf(tt::Leaf::Literal(tt::Literal{  symbol: text, kind: tt::LitKind::Str | tt::LitKind::StrRaw(_) , ..}))) => Some(text.as_str()),
                _ => None
            }
        })
    }
}

fn any_has_attrs<'db>(
    db: &(dyn DefDatabase + 'db),
    id: impl Lookup<Database = dyn DefDatabase, Data = impl HasSource<Value = impl ast::HasAttrs>>,
) -> InFile<ast::AnyHasAttrs> {
    id.lookup(db).source(db).map(ast::AnyHasAttrs::new)
}

fn attrs_from_ast_id_loc<'db, N: AstIdNode + HasAttrs>(
    db: &(dyn DefDatabase + 'db),
    lookup: impl Lookup<Database = dyn DefDatabase, Data = impl AstIdLoc<Ast = N> + HasModule>,
) -> Attrs {
    let loc = lookup.lookup(db);
    let source = loc.source(db);
    let span_map = db.span_map(source.file_id);
    let cfg_options = loc.krate(db).cfg_options(db);
    Attrs(RawAttrs::new_expanded(db, &source.value, span_map.as_ref(), cfg_options))
}

pub(crate) fn fields_attrs_source_map(
    db: &dyn DefDatabase,
    def: VariantId,
) -> Arc<ArenaMap<LocalFieldId, AstPtr<Either<ast::TupleField, ast::RecordField>>>> {
    let mut res = ArenaMap::default();
    let child_source = def.child_source(db);

    for (idx, variant) in child_source.value.iter() {
        res.insert(
            idx,
            variant
                .as_ref()
                .either(|l| AstPtr::new(l).wrap_left(), |r| AstPtr::new(r).wrap_right()),
        );
    }

    Arc::new(res)
}

#[cfg(test)]
mod tests {
    //! This module contains tests for doc-expression parsing.
    //! Currently, it tests `#[doc(hidden)]` and `#[doc(alias)]`.

    use intern::Symbol;
    use span::EditionedFileId;
    use triomphe::Arc;

    use hir_expand::span_map::{RealSpanMap, SpanMap};
    use span::FileId;
    use syntax::{AstNode, TextRange, ast};
    use syntax_bridge::{DocCommentDesugarMode, syntax_node_to_token_tree};

    use crate::attr::{DocAtom, DocExpr};

    fn assert_parse_result(input: &str, expected: DocExpr) {
        let source_file = ast::SourceFile::parse(input, span::Edition::CURRENT).ok().unwrap();
        let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
        let map = SpanMap::RealSpanMap(Arc::new(RealSpanMap::absolute(
            EditionedFileId::current_edition(FileId::from_raw(0)),
        )));
        let tt = syntax_node_to_token_tree(
            tt.syntax(),
            map.as_ref(),
            map.span_for_range(TextRange::empty(0.into())),
            DocCommentDesugarMode::ProcMacro,
        );
        let cfg = DocExpr::parse(&tt);
        assert_eq!(cfg, expected);
    }

    #[test]
    fn test_doc_expr_parser() {
        assert_parse_result("#![doc(hidden)]", DocAtom::Flag(Symbol::intern("hidden")).into());

        assert_parse_result(
            r#"#![doc(alias = "foo")]"#,
            DocAtom::KeyValue { key: Symbol::intern("alias"), value: Symbol::intern("foo") }.into(),
        );

        assert_parse_result(
            r#"#![doc(alias("foo"))]"#,
            DocExpr::Alias([Symbol::intern("foo")].into()),
        );
        assert_parse_result(
            r#"#![doc(alias("foo", "bar", "baz"))]"#,
            DocExpr::Alias(
                [Symbol::intern("foo"), Symbol::intern("bar"), Symbol::intern("baz")].into(),
            ),
        );

        assert_parse_result(
            r#"
        #[doc(alias("Bar", "Qux"))]
        struct Foo;"#,
            DocExpr::Alias([Symbol::intern("Bar"), Symbol::intern("Qux")].into()),
        );
    }
}
