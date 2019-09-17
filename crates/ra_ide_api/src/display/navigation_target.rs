use hir::{AssocItem, FieldSource, HasSource, ModuleSource};
use ra_db::{FileId, SourceDatabase};
use ra_syntax::{
    algo::visit::{visitor, Visitor},
    ast::{self, DocCommentsOwner},
    AstNode, AstPtr, SmolStr,
    SyntaxKind::{self, NAME},
    SyntaxNode, TextRange,
};

use super::short_label::ShortLabel;
use crate::{db::RootDatabase, FileSymbol};

/// `NavigationTarget` represents and element in the editor's UI which you can
/// click on to navigate to a particular piece of code.
///
/// Typically, a `NavigationTarget` corresponds to some element in the source
/// code, like a function or a struct, but this is not strictly required.
#[derive(Debug, Clone)]
pub struct NavigationTarget {
    file_id: FileId,
    name: SmolStr,
    kind: SyntaxKind,
    full_range: TextRange,
    focus_range: Option<TextRange>,
    container_name: Option<SmolStr>,
    description: Option<String>,
    docs: Option<String>,
}

impl NavigationTarget {
    /// When `focus_range` is specified, returns it. otherwise
    /// returns `full_range`
    pub fn range(&self) -> TextRange {
        self.focus_range.unwrap_or(self.full_range)
    }

    pub fn name(&self) -> &SmolStr {
        &self.name
    }

    pub fn container_name(&self) -> Option<&SmolStr> {
        self.container_name.as_ref()
    }

    pub fn kind(&self) -> SyntaxKind {
        self.kind
    }

    pub fn file_id(&self) -> FileId {
        self.file_id
    }

    pub fn full_range(&self) -> TextRange {
        self.full_range
    }

    pub fn docs(&self) -> Option<&str> {
        self.docs.as_ref().map(String::as_str)
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_ref().map(String::as_str)
    }

    /// A "most interesting" range withing the `full_range`.
    ///
    /// Typically, `full_range` is the whole syntax node,
    /// including doc comments, and `focus_range` is the range of the identifier.
    pub fn focus_range(&self) -> Option<TextRange> {
        self.focus_range
    }

    pub(crate) fn from_bind_pat(file_id: FileId, pat: &ast::BindPat) -> NavigationTarget {
        NavigationTarget::from_named(file_id, pat, None, None)
    }

    pub(crate) fn from_symbol(db: &RootDatabase, symbol: FileSymbol) -> NavigationTarget {
        NavigationTarget {
            file_id: symbol.file_id,
            name: symbol.name.clone(),
            kind: symbol.ptr.kind(),
            full_range: symbol.ptr.range(),
            focus_range: symbol.name_range,
            container_name: symbol.container_name.clone(),
            description: description_from_symbol(db, &symbol),
            docs: docs_from_symbol(db, &symbol),
        }
    }

    pub(crate) fn from_pat(
        db: &RootDatabase,
        file_id: FileId,
        pat: AstPtr<ast::BindPat>,
    ) -> NavigationTarget {
        let parse = db.parse(file_id);
        let pat = pat.to_node(parse.tree().syntax());
        NavigationTarget::from_bind_pat(file_id, &pat)
    }

    pub(crate) fn from_self_param(
        file_id: FileId,
        par: AstPtr<ast::SelfParam>,
    ) -> NavigationTarget {
        let (name, full_range) = ("self".into(), par.syntax_node_ptr().range());

        NavigationTarget {
            file_id,
            name,
            full_range,
            focus_range: None,
            kind: NAME,
            container_name: None,
            description: None, //< No document node for SelfParam
            docs: None,        //< No document node for SelfParam
        }
    }

    pub(crate) fn from_module(db: &RootDatabase, module: hir::Module) -> NavigationTarget {
        let src = module.definition_source(db);
        let file_id = src.file_id.as_original_file();
        let name = module.name(db).map(|it| it.to_string().into()).unwrap_or_default();
        match src.ast {
            ModuleSource::SourceFile(node) => {
                NavigationTarget::from_syntax(file_id, name, None, node.syntax(), None, None)
            }
            ModuleSource::Module(node) => NavigationTarget::from_syntax(
                file_id,
                name,
                None,
                node.syntax(),
                node.doc_comment_text(),
                node.short_label(),
            ),
        }
    }

    pub(crate) fn from_module_to_decl(db: &RootDatabase, module: hir::Module) -> NavigationTarget {
        let name = module.name(db).map(|it| it.to_string().into()).unwrap_or_default();
        if let Some(src) = module.declaration_source(db) {
            let file_id = src.file_id.as_original_file();
            return NavigationTarget::from_syntax(
                file_id,
                name,
                None,
                src.ast.syntax(),
                src.ast.doc_comment_text(),
                src.ast.short_label(),
            );
        }
        NavigationTarget::from_module(db, module)
    }

    pub(crate) fn from_field(db: &RootDatabase, field: hir::StructField) -> NavigationTarget {
        let src = field.source(db);
        let file_id = src.file_id.original_file(db);
        match src.ast {
            FieldSource::Named(it) => {
                NavigationTarget::from_named(file_id, &it, it.doc_comment_text(), it.short_label())
            }
            FieldSource::Pos(it) => {
                NavigationTarget::from_syntax(file_id, "".into(), None, it.syntax(), None, None)
            }
        }
    }

    pub(crate) fn from_def_source<A, D>(db: &RootDatabase, def: D) -> NavigationTarget
    where
        D: HasSource<Ast = A>,
        A: ast::DocCommentsOwner + ast::NameOwner + ShortLabel,
    {
        let src = def.source(db);
        NavigationTarget::from_named(
            src.file_id.original_file(db),
            &src.ast,
            src.ast.doc_comment_text(),
            src.ast.short_label(),
        )
    }

    pub(crate) fn from_adt_def(db: &RootDatabase, adt_def: hir::Adt) -> NavigationTarget {
        match adt_def {
            hir::Adt::Struct(it) => NavigationTarget::from_def_source(db, it),
            hir::Adt::Union(it) => NavigationTarget::from_def_source(db, it),
            hir::Adt::Enum(it) => NavigationTarget::from_def_source(db, it),
        }
    }

    pub(crate) fn from_def(
        db: &RootDatabase,
        module_def: hir::ModuleDef,
    ) -> Option<NavigationTarget> {
        let nav = match module_def {
            hir::ModuleDef::Module(module) => NavigationTarget::from_module(db, module),
            hir::ModuleDef::Function(func) => NavigationTarget::from_def_source(db, func),
            hir::ModuleDef::Adt(it) => NavigationTarget::from_adt_def(db, it),
            hir::ModuleDef::Const(it) => NavigationTarget::from_def_source(db, it),
            hir::ModuleDef::Static(it) => NavigationTarget::from_def_source(db, it),
            hir::ModuleDef::EnumVariant(it) => NavigationTarget::from_def_source(db, it),
            hir::ModuleDef::Trait(it) => NavigationTarget::from_def_source(db, it),
            hir::ModuleDef::TypeAlias(it) => NavigationTarget::from_def_source(db, it),
            hir::ModuleDef::BuiltinType(..) => {
                return None;
            }
        };
        Some(nav)
    }

    pub(crate) fn from_impl_block(
        db: &RootDatabase,
        impl_block: hir::ImplBlock,
    ) -> NavigationTarget {
        let src = impl_block.source(db);
        NavigationTarget::from_syntax(
            src.file_id.as_original_file(),
            "impl".into(),
            None,
            src.ast.syntax(),
            None,
            None,
        )
    }

    pub(crate) fn from_assoc_item(
        db: &RootDatabase,
        assoc_item: hir::AssocItem,
    ) -> NavigationTarget {
        match assoc_item {
            AssocItem::Function(it) => NavigationTarget::from_def_source(db, it),
            AssocItem::Const(it) => NavigationTarget::from_def_source(db, it),
            AssocItem::TypeAlias(it) => NavigationTarget::from_def_source(db, it),
        }
    }

    pub(crate) fn from_macro_def(db: &RootDatabase, macro_call: hir::MacroDef) -> NavigationTarget {
        let src = macro_call.source(db);
        log::debug!("nav target {:#?}", src.ast.syntax());
        NavigationTarget::from_named(
            src.file_id.original_file(db),
            &src.ast,
            src.ast.doc_comment_text(),
            None,
        )
    }

    #[cfg(test)]
    pub(crate) fn assert_match(&self, expected: &str) {
        let actual = self.debug_render();
        test_utils::assert_eq_text!(expected.trim(), actual.trim(),);
    }

    #[cfg(test)]
    pub(crate) fn debug_render(&self) -> String {
        let mut buf = format!(
            "{} {:?} {:?} {:?}",
            self.name(),
            self.kind(),
            self.file_id(),
            self.full_range()
        );
        if let Some(focus_range) = self.focus_range() {
            buf.push_str(&format!(" {:?}", focus_range))
        }
        if let Some(container_name) = self.container_name() {
            buf.push_str(&format!(" {}", container_name))
        }
        buf
    }

    /// Allows `NavigationTarget` to be created from a `NameOwner`
    pub(crate) fn from_named(
        file_id: FileId,
        node: &impl ast::NameOwner,
        docs: Option<String>,
        description: Option<String>,
    ) -> NavigationTarget {
        //FIXME: use `_` instead of empty string
        let name = node.name().map(|it| it.text().clone()).unwrap_or_default();
        let focus_range = node.name().map(|it| it.syntax().text_range());
        NavigationTarget::from_syntax(file_id, name, focus_range, node.syntax(), docs, description)
    }

    fn from_syntax(
        file_id: FileId,
        name: SmolStr,
        focus_range: Option<TextRange>,
        node: &SyntaxNode,
        docs: Option<String>,
        description: Option<String>,
    ) -> NavigationTarget {
        NavigationTarget {
            file_id,
            name,
            kind: node.kind(),
            full_range: node.text_range(),
            focus_range,
            // ptr: Some(LocalSyntaxPtr::new(node)),
            container_name: None,
            description,
            docs,
        }
    }
}

pub(crate) fn docs_from_symbol(db: &RootDatabase, symbol: &FileSymbol) -> Option<String> {
    let parse = db.parse(symbol.file_id);
    let node = symbol.ptr.to_node(parse.tree().syntax()).to_owned();

    visitor()
        .visit(|it: ast::FnDef| it.doc_comment_text())
        .visit(|it: ast::StructDef| it.doc_comment_text())
        .visit(|it: ast::EnumDef| it.doc_comment_text())
        .visit(|it: ast::TraitDef| it.doc_comment_text())
        .visit(|it: ast::Module| it.doc_comment_text())
        .visit(|it: ast::TypeAliasDef| it.doc_comment_text())
        .visit(|it: ast::ConstDef| it.doc_comment_text())
        .visit(|it: ast::StaticDef| it.doc_comment_text())
        .visit(|it: ast::RecordFieldDef| it.doc_comment_text())
        .visit(|it: ast::EnumVariant| it.doc_comment_text())
        .visit(|it: ast::MacroCall| it.doc_comment_text())
        .accept(&node)?
}

/// Get a description of a symbol.
///
/// e.g. `struct Name`, `enum Name`, `fn Name`
pub(crate) fn description_from_symbol(db: &RootDatabase, symbol: &FileSymbol) -> Option<String> {
    let parse = db.parse(symbol.file_id);
    let node = symbol.ptr.to_node(parse.tree().syntax()).to_owned();

    visitor()
        .visit(|node: ast::FnDef| node.short_label())
        .visit(|node: ast::StructDef| node.short_label())
        .visit(|node: ast::EnumDef| node.short_label())
        .visit(|node: ast::TraitDef| node.short_label())
        .visit(|node: ast::Module| node.short_label())
        .visit(|node: ast::TypeAliasDef| node.short_label())
        .visit(|node: ast::ConstDef| node.short_label())
        .visit(|node: ast::StaticDef| node.short_label())
        .visit(|node: ast::RecordFieldDef| node.short_label())
        .visit(|node: ast::EnumVariant| node.short_label())
        .accept(&node)?
}
