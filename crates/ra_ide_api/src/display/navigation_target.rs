use ra_db::{FileId, SourceDatabase};
use ra_syntax::{
    SyntaxNode, AstNode, SmolStr, TextRange, TreeArc, AstPtr,
    SyntaxKind::{self, NAME},
    ast::{self, NameOwner, VisibilityOwner, TypeAscriptionOwner},
    algo::visit::{visitor, Visitor},
};
use hir::{ModuleSource, FieldSource, ImplItem, Either};

use crate::{FileSymbol, db::RootDatabase};

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

    /// A "most interesting" range withing the `full_range`.
    ///
    /// Typically, `full_range` is the whole syntax node,
    /// including doc comments, and `focus_range` is the range of the identifier.
    pub fn focus_range(&self) -> Option<TextRange> {
        self.focus_range
    }

    pub(crate) fn from_bind_pat(file_id: FileId, pat: &ast::BindPat) -> NavigationTarget {
        NavigationTarget::from_named(file_id, pat)
    }

    pub(crate) fn from_symbol(symbol: FileSymbol) -> NavigationTarget {
        NavigationTarget {
            file_id: symbol.file_id,
            name: symbol.name.clone(),
            kind: symbol.ptr.kind(),
            full_range: symbol.ptr.range(),
            focus_range: symbol.name_range,
            container_name: symbol.container_name.clone(),
        }
    }

    pub(crate) fn from_pat(
        db: &RootDatabase,
        file_id: FileId,
        pat: Either<AstPtr<ast::Pat>, AstPtr<ast::SelfParam>>,
    ) -> NavigationTarget {
        let file = db.parse(file_id);
        let (name, full_range) = match pat {
            Either::A(pat) => match pat.to_node(&file).kind() {
                ast::PatKind::BindPat(pat) => {
                    return NavigationTarget::from_bind_pat(file_id, &pat)
                }
                _ => ("_".into(), pat.syntax_node_ptr().range()),
            },
            Either::B(slf) => ("self".into(), slf.syntax_node_ptr().range()),
        };
        NavigationTarget {
            file_id,
            name,
            full_range,
            focus_range: None,
            kind: NAME,
            container_name: None,
        }
    }

    pub(crate) fn from_module(db: &RootDatabase, module: hir::Module) -> NavigationTarget {
        let (file_id, source) = module.definition_source(db);
        let file_id = file_id.as_original_file();
        let name = module.name(db).map(|it| it.to_string().into()).unwrap_or_default();
        match source {
            ModuleSource::SourceFile(node) => {
                NavigationTarget::from_syntax(file_id, name, None, node.syntax())
            }
            ModuleSource::Module(node) => {
                NavigationTarget::from_syntax(file_id, name, None, node.syntax())
            }
        }
    }

    pub(crate) fn from_module_to_decl(db: &RootDatabase, module: hir::Module) -> NavigationTarget {
        let name = module.name(db).map(|it| it.to_string().into()).unwrap_or_default();
        if let Some((file_id, source)) = module.declaration_source(db) {
            let file_id = file_id.as_original_file();
            return NavigationTarget::from_syntax(file_id, name, None, source.syntax());
        }
        NavigationTarget::from_module(db, module)
    }

    pub(crate) fn from_function(db: &RootDatabase, func: hir::Function) -> NavigationTarget {
        let (file_id, fn_def) = func.source(db);
        NavigationTarget::from_named(file_id.original_file(db), &*fn_def)
    }

    pub(crate) fn from_field(db: &RootDatabase, field: hir::StructField) -> NavigationTarget {
        let (file_id, field) = field.source(db);
        let file_id = file_id.original_file(db);
        match field {
            FieldSource::Named(it) => NavigationTarget::from_named(file_id, &*it),
            FieldSource::Pos(it) => {
                NavigationTarget::from_syntax(file_id, "".into(), None, it.syntax())
            }
        }
    }

    pub(crate) fn from_adt_def(db: &RootDatabase, adt_def: hir::AdtDef) -> NavigationTarget {
        match adt_def {
            hir::AdtDef::Struct(s) => {
                let (file_id, node) = s.source(db);
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
            hir::AdtDef::Enum(s) => {
                let (file_id, node) = s.source(db);
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
        }
    }

    pub(crate) fn from_def(db: &RootDatabase, module_def: hir::ModuleDef) -> NavigationTarget {
        match module_def {
            hir::ModuleDef::Module(module) => NavigationTarget::from_module(db, module),
            hir::ModuleDef::Function(func) => NavigationTarget::from_function(db, func),
            hir::ModuleDef::Struct(s) => {
                let (file_id, node) = s.source(db);
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
            hir::ModuleDef::Const(s) => {
                let (file_id, node) = s.source(db);
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
            hir::ModuleDef::Static(s) => {
                let (file_id, node) = s.source(db);
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
            hir::ModuleDef::Enum(e) => {
                let (file_id, node) = e.source(db);
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
            hir::ModuleDef::EnumVariant(var) => {
                let (file_id, node) = var.source(db);
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
            hir::ModuleDef::Trait(e) => {
                let (file_id, node) = e.source(db);
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
            hir::ModuleDef::TypeAlias(e) => {
                let (file_id, node) = e.source(db);
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
        }
    }

    pub(crate) fn from_impl_block(
        db: &RootDatabase,
        impl_block: hir::ImplBlock,
    ) -> NavigationTarget {
        let (file_id, node) = impl_block.source(db);
        NavigationTarget::from_syntax(
            file_id.as_original_file(),
            "impl".into(),
            None,
            node.syntax(),
        )
    }

    pub(crate) fn from_impl_item(db: &RootDatabase, impl_item: hir::ImplItem) -> NavigationTarget {
        match impl_item {
            ImplItem::Method(f) => NavigationTarget::from_function(db, f),
            ImplItem::Const(c) => {
                let (file_id, node) = c.source(db);
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
            ImplItem::TypeAlias(a) => {
                let (file_id, node) = a.source(db);
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
        }
    }

    pub(crate) fn from_macro_def(
        db: &RootDatabase,
        macro_call: hir::MacroByExampleDef,
    ) -> NavigationTarget {
        let (file_id, node) = macro_call.source(db);
        log::debug!("nav target {}", node.syntax().debug_dump());
        NavigationTarget::from_named(file_id.original_file(db), &*node)
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
    pub(crate) fn from_named(file_id: FileId, node: &impl ast::NameOwner) -> NavigationTarget {
        //FIXME: use `_` instead of empty string
        let name = node.name().map(|it| it.text().clone()).unwrap_or_default();
        let focus_range = node.name().map(|it| it.syntax().range());
        NavigationTarget::from_syntax(file_id, name, focus_range, node.syntax())
    }

    fn from_syntax(
        file_id: FileId,
        name: SmolStr,
        focus_range: Option<TextRange>,
        node: &SyntaxNode,
    ) -> NavigationTarget {
        NavigationTarget {
            file_id,
            name,
            kind: node.kind(),
            full_range: node.range(),
            focus_range,
            // ptr: Some(LocalSyntaxPtr::new(node)),
            container_name: None,
        }
    }

    pub(crate) fn node(&self, db: &RootDatabase) -> Option<TreeArc<SyntaxNode>> {
        let source_file = db.parse(self.file_id());
        let source_file = source_file.syntax();
        let node = source_file
            .descendants()
            .find(|node| node.kind() == self.kind() && node.range() == self.full_range())?
            .to_owned();
        Some(node)
    }

    pub(crate) fn docs(&self, db: &RootDatabase) -> Option<String> {
        let node = self.node(db)?;
        fn doc_comments<N: ast::DocCommentsOwner>(node: &N) -> Option<String> {
            node.doc_comment_text()
        }

        visitor()
            .visit(doc_comments::<ast::FnDef>)
            .visit(doc_comments::<ast::StructDef>)
            .visit(doc_comments::<ast::EnumDef>)
            .visit(doc_comments::<ast::TraitDef>)
            .visit(doc_comments::<ast::Module>)
            .visit(doc_comments::<ast::TypeAliasDef>)
            .visit(doc_comments::<ast::ConstDef>)
            .visit(doc_comments::<ast::StaticDef>)
            .visit(doc_comments::<ast::NamedFieldDef>)
            .visit(doc_comments::<ast::EnumVariant>)
            .visit(doc_comments::<ast::MacroCall>)
            .accept(&node)?
    }

    /// Get a description of this node.
    ///
    /// e.g. `struct Name`, `enum Name`, `fn Name`
    pub(crate) fn description(&self, db: &RootDatabase) -> Option<String> {
        // FIXME: After type inference is done, add type information to improve the output
        let node = self.node(db)?;

        fn visit_ascribed_node<T>(node: &T, prefix: &str) -> Option<String>
        where
            T: NameOwner + VisibilityOwner + TypeAscriptionOwner,
        {
            let mut string = visit_node(node, prefix)?;

            if let Some(type_ref) = node.ascribed_type() {
                string.push_str(": ");
                type_ref.syntax().text().push_to(&mut string);
            }

            Some(string)
        }

        fn visit_node<T>(node: &T, label: &str) -> Option<String>
        where
            T: NameOwner + VisibilityOwner,
        {
            let mut string =
                node.visibility().map(|v| format!("{} ", v.syntax().text())).unwrap_or_default();
            string.push_str(label);
            string.push_str(node.name()?.text().as_str());
            Some(string)
        }

        visitor()
            .visit(|node: &ast::FnDef| Some(crate::display::function_label(node)))
            .visit(|node: &ast::StructDef| visit_node(node, "struct "))
            .visit(|node: &ast::EnumDef| visit_node(node, "enum "))
            .visit(|node: &ast::TraitDef| visit_node(node, "trait "))
            .visit(|node: &ast::Module| visit_node(node, "mod "))
            .visit(|node: &ast::TypeAliasDef| visit_node(node, "type "))
            .visit(|node: &ast::ConstDef| visit_ascribed_node(node, "const "))
            .visit(|node: &ast::StaticDef| visit_ascribed_node(node, "static "))
            .visit(|node: &ast::NamedFieldDef| visit_ascribed_node(node, ""))
            .visit(|node: &ast::EnumVariant| Some(node.name()?.text().to_string()))
            .accept(&node)?
    }
}
