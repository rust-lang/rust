//! FIXME: write short doc here

use base_db::{FileId, SourceDatabase};
use either::Either;
use hir::{original_range, AssocItem, FieldSource, HasSource, InFile, ModuleSource};
use ide_db::{defs::Definition, RootDatabase};
use syntax::{
    ast::{self, DocCommentsOwner, NameOwner},
    match_ast, AstNode, SmolStr,
    SyntaxKind::{self, IDENT_PAT, TYPE_PARAM},
    TextRange,
};

use crate::FileSymbol;

use super::short_label::ShortLabel;

/// `NavigationTarget` represents and element in the editor's UI which you can
/// click on to navigate to a particular piece of code.
///
/// Typically, a `NavigationTarget` corresponds to some element in the source
/// code, like a function or a struct, but this is not strictly required.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NavigationTarget {
    pub file_id: FileId,
    /// Range which encompasses the whole element.
    ///
    /// Should include body, doc comments, attributes, etc.
    ///
    /// Clients should use this range to answer "is the cursor inside the
    /// element?" question.
    pub full_range: TextRange,
    /// A "most interesting" range withing the `full_range`.
    ///
    /// Typically, `full_range` is the whole syntax node, including doc
    /// comments, and `focus_range` is the range of the identifier. "Most
    /// interesting" range within the full range, typically the range of
    /// identifier.
    ///
    /// Clients should place the cursor on this range when navigating to this target.
    pub focus_range: Option<TextRange>,
    pub name: SmolStr,
    pub kind: SyntaxKind,
    pub container_name: Option<SmolStr>,
    pub description: Option<String>,
    pub docs: Option<String>,
}

pub(crate) trait ToNav {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget;
}

pub(crate) trait TryToNav {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget>;
}

impl NavigationTarget {
    pub fn focus_or_full_range(&self) -> TextRange {
        self.focus_range.unwrap_or(self.full_range)
    }

    pub(crate) fn from_module_to_decl(db: &RootDatabase, module: hir::Module) -> NavigationTarget {
        let name = module.name(db).map(|it| it.to_string().into()).unwrap_or_default();
        if let Some(src) = module.declaration_source(db) {
            let frange = original_range(db, src.as_ref().map(|it| it.syntax()));
            let mut res = NavigationTarget::from_syntax(
                frange.file_id,
                name,
                None,
                frange.range,
                src.value.syntax().kind(),
            );
            res.docs = src.value.doc_comment_text();
            res.description = src.value.short_label();
            return res;
        }
        module.to_nav(db)
    }

    #[cfg(test)]
    pub(crate) fn assert_match(&self, expected: &str) {
        let actual = self.debug_render();
        test_utils::assert_eq_text!(expected.trim(), actual.trim(),);
    }

    #[cfg(test)]
    pub(crate) fn debug_render(&self) -> String {
        let mut buf =
            format!("{} {:?} {:?} {:?}", self.name, self.kind, self.file_id, self.full_range);
        if let Some(focus_range) = self.focus_range {
            buf.push_str(&format!(" {:?}", focus_range))
        }
        if let Some(container_name) = &self.container_name {
            buf.push_str(&format!(" {}", container_name))
        }
        buf
    }

    /// Allows `NavigationTarget` to be created from a `NameOwner`
    pub(crate) fn from_named(
        db: &RootDatabase,
        node: InFile<&dyn ast::NameOwner>,
    ) -> NavigationTarget {
        let name =
            node.value.name().map(|it| it.text().clone()).unwrap_or_else(|| SmolStr::new("_"));
        let focus_range =
            node.value.name().map(|it| original_range(db, node.with_value(it.syntax())).range);
        let frange = original_range(db, node.map(|it| it.syntax()));

        NavigationTarget::from_syntax(
            frange.file_id,
            name,
            focus_range,
            frange.range,
            node.value.syntax().kind(),
        )
    }

    /// Allows `NavigationTarget` to be created from a `DocCommentsOwner` and a `NameOwner`
    pub(crate) fn from_doc_commented(
        db: &RootDatabase,
        named: InFile<&dyn ast::NameOwner>,
        node: InFile<&dyn ast::DocCommentsOwner>,
    ) -> NavigationTarget {
        let name =
            named.value.name().map(|it| it.text().clone()).unwrap_or_else(|| SmolStr::new("_"));
        let frange = original_range(db, node.map(|it| it.syntax()));

        NavigationTarget::from_syntax(
            frange.file_id,
            name,
            None,
            frange.range,
            node.value.syntax().kind(),
        )
    }

    fn from_syntax(
        file_id: FileId,
        name: SmolStr,
        focus_range: Option<TextRange>,
        full_range: TextRange,
        kind: SyntaxKind,
    ) -> NavigationTarget {
        NavigationTarget {
            file_id,
            name,
            kind,
            full_range,
            focus_range,
            container_name: None,
            description: None,
            docs: None,
        }
    }
}

impl ToNav for FileSymbol {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        NavigationTarget {
            file_id: self.file_id,
            name: self.name.clone(),
            kind: self.kind,
            full_range: self.range,
            focus_range: self.name_range,
            container_name: self.container_name.clone(),
            description: description_from_symbol(db, self),
            docs: docs_from_symbol(db, self),
        }
    }
}

impl TryToNav for Definition {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        match self {
            Definition::Macro(it) => Some(it.to_nav(db)),
            Definition::Field(it) => Some(it.to_nav(db)),
            Definition::ModuleDef(it) => it.try_to_nav(db),
            Definition::SelfType(it) => Some(it.to_nav(db)),
            Definition::Local(it) => Some(it.to_nav(db)),
            Definition::TypeParam(it) => Some(it.to_nav(db)),
        }
    }
}

impl TryToNav for hir::ModuleDef {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        let res = match self {
            hir::ModuleDef::Module(it) => it.to_nav(db),
            hir::ModuleDef::Function(it) => it.to_nav(db),
            hir::ModuleDef::Adt(it) => it.to_nav(db),
            hir::ModuleDef::EnumVariant(it) => it.to_nav(db),
            hir::ModuleDef::Const(it) => it.to_nav(db),
            hir::ModuleDef::Static(it) => it.to_nav(db),
            hir::ModuleDef::Trait(it) => it.to_nav(db),
            hir::ModuleDef::TypeAlias(it) => it.to_nav(db),
            hir::ModuleDef::BuiltinType(_) => return None,
        };
        Some(res)
    }
}

pub(crate) trait ToNavFromAst {}
impl ToNavFromAst for hir::Function {}
impl ToNavFromAst for hir::Const {}
impl ToNavFromAst for hir::Static {}
impl ToNavFromAst for hir::Struct {}
impl ToNavFromAst for hir::Enum {}
impl ToNavFromAst for hir::EnumVariant {}
impl ToNavFromAst for hir::Union {}
impl ToNavFromAst for hir::TypeAlias {}
impl ToNavFromAst for hir::Trait {}

impl<D> ToNav for D
where
    D: HasSource + ToNavFromAst + Copy,
    D::Ast: ast::DocCommentsOwner + ast::NameOwner + ShortLabel,
{
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let src = self.source(db);
        let mut res =
            NavigationTarget::from_named(db, src.as_ref().map(|it| it as &dyn ast::NameOwner));
        res.docs = src.value.doc_comment_text();
        res.description = src.value.short_label();
        res
    }
}

impl ToNav for hir::Module {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let src = self.definition_source(db);
        let name = self.name(db).map(|it| it.to_string().into()).unwrap_or_default();
        let (syntax, focus) = match &src.value {
            ModuleSource::SourceFile(node) => (node.syntax(), None),
            ModuleSource::Module(node) => {
                (node.syntax(), node.name().map(|it| it.syntax().text_range()))
            }
        };
        let frange = original_range(db, src.with_value(syntax));
        NavigationTarget::from_syntax(frange.file_id, name, focus, frange.range, syntax.kind())
    }
}

impl ToNav for hir::ImplDef {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let src = self.source(db);
        let derive_attr = self.is_builtin_derive(db);
        let frange = if let Some(item) = &derive_attr {
            original_range(db, item.syntax())
        } else {
            original_range(db, src.as_ref().map(|it| it.syntax()))
        };
        let focus_range = if derive_attr.is_some() {
            None
        } else {
            src.value.self_ty().map(|ty| original_range(db, src.with_value(ty.syntax())).range)
        };

        NavigationTarget::from_syntax(
            frange.file_id,
            "impl".into(),
            focus_range,
            frange.range,
            src.value.syntax().kind(),
        )
    }
}

impl ToNav for hir::Field {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let src = self.source(db);

        match &src.value {
            FieldSource::Named(it) => {
                let mut res = NavigationTarget::from_named(db, src.with_value(it));
                res.docs = it.doc_comment_text();
                res.description = it.short_label();
                res
            }
            FieldSource::Pos(it) => {
                let frange = original_range(db, src.with_value(it.syntax()));
                NavigationTarget::from_syntax(
                    frange.file_id,
                    "".into(),
                    None,
                    frange.range,
                    it.syntax().kind(),
                )
            }
        }
    }
}

impl ToNav for hir::MacroDef {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let src = self.source(db);
        log::debug!("nav target {:#?}", src.value.syntax());
        let mut res =
            NavigationTarget::from_named(db, src.as_ref().map(|it| it as &dyn ast::NameOwner));
        res.docs = src.value.doc_comment_text();
        res
    }
}

impl ToNav for hir::Adt {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        match self {
            hir::Adt::Struct(it) => it.to_nav(db),
            hir::Adt::Union(it) => it.to_nav(db),
            hir::Adt::Enum(it) => it.to_nav(db),
        }
    }
}

impl ToNav for hir::AssocItem {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        match self {
            AssocItem::Function(it) => it.to_nav(db),
            AssocItem::Const(it) => it.to_nav(db),
            AssocItem::TypeAlias(it) => it.to_nav(db),
        }
    }
}

impl ToNav for hir::Local {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let src = self.source(db);
        let node = match &src.value {
            Either::Left(bind_pat) => {
                bind_pat.name().map_or_else(|| bind_pat.syntax().clone(), |it| it.syntax().clone())
            }
            Either::Right(it) => it.syntax().clone(),
        };
        let full_range = original_range(db, src.with_value(&node));
        let name = match self.name(db) {
            Some(it) => it.to_string().into(),
            None => "".into(),
        };
        NavigationTarget {
            file_id: full_range.file_id,
            name,
            kind: IDENT_PAT,
            full_range: full_range.range,
            focus_range: None,
            container_name: None,
            description: None,
            docs: None,
        }
    }
}

impl ToNav for hir::TypeParam {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let src = self.source(db);
        let full_range = match &src.value {
            Either::Left(it) => it.syntax().text_range(),
            Either::Right(it) => it.syntax().text_range(),
        };
        let focus_range = match &src.value {
            Either::Left(_) => None,
            Either::Right(it) => it.name().map(|it| it.syntax().text_range()),
        };
        NavigationTarget {
            file_id: src.file_id.original_file(db),
            name: self.name(db).to_string().into(),
            kind: TYPE_PARAM,
            full_range,
            focus_range,
            container_name: None,
            description: None,
            docs: None,
        }
    }
}

pub(crate) fn docs_from_symbol(db: &RootDatabase, symbol: &FileSymbol) -> Option<String> {
    let parse = db.parse(symbol.file_id);
    let node = symbol.ptr.to_node(parse.tree().syntax());

    match_ast! {
        match node {
            ast::Fn(it) => it.doc_comment_text(),
            ast::Struct(it) => it.doc_comment_text(),
            ast::Enum(it) => it.doc_comment_text(),
            ast::Trait(it) => it.doc_comment_text(),
            ast::Module(it) => it.doc_comment_text(),
            ast::TypeAlias(it) => it.doc_comment_text(),
            ast::Const(it) => it.doc_comment_text(),
            ast::Static(it) => it.doc_comment_text(),
            ast::RecordField(it) => it.doc_comment_text(),
            ast::Variant(it) => it.doc_comment_text(),
            ast::MacroCall(it) => it.doc_comment_text(),
            _ => None,
        }
    }
}

/// Get a description of a symbol.
///
/// e.g. `struct Name`, `enum Name`, `fn Name`
pub(crate) fn description_from_symbol(db: &RootDatabase, symbol: &FileSymbol) -> Option<String> {
    let parse = db.parse(symbol.file_id);
    let node = symbol.ptr.to_node(parse.tree().syntax());

    match_ast! {
        match node {
            ast::Fn(it) => it.short_label(),
            ast::Struct(it) => it.short_label(),
            ast::Enum(it) => it.short_label(),
            ast::Trait(it) => it.short_label(),
            ast::Module(it) => it.short_label(),
            ast::TypeAlias(it) => it.short_label(),
            ast::Const(it) => it.short_label(),
            ast::Static(it) => it.short_label(),
            ast::RecordField(it) => it.short_label(),
            ast::Variant(it) => it.short_label(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::{mock_analysis::single_file, Query};

    #[test]
    fn test_nav_for_symbol() {
        let (analysis, _) = single_file(
            r#"
enum FooInner { }
fn foo() { enum FooInner { } }
"#,
        );

        let navs = analysis.symbol_search(Query::new("FooInner".to_string())).unwrap();
        expect![[r#"
            [
                NavigationTarget {
                    file_id: FileId(
                        1,
                    ),
                    full_range: 0..17,
                    focus_range: Some(
                        5..13,
                    ),
                    name: "FooInner",
                    kind: ENUM,
                    container_name: None,
                    description: Some(
                        "enum FooInner",
                    ),
                    docs: None,
                },
                NavigationTarget {
                    file_id: FileId(
                        1,
                    ),
                    full_range: 29..46,
                    focus_range: Some(
                        34..42,
                    ),
                    name: "FooInner",
                    kind: ENUM,
                    container_name: Some(
                        "foo",
                    ),
                    description: Some(
                        "enum FooInner",
                    ),
                    docs: None,
                },
            ]
        "#]]
        .assert_debug_eq(&navs);
    }

    #[test]
    fn test_world_symbols_are_case_sensitive() {
        let (analysis, _) = single_file(
            r#"
fn foo() {}
struct Foo;
"#,
        );

        let navs = analysis.symbol_search(Query::new("foo".to_string())).unwrap();
        assert_eq!(navs.len(), 2)
    }
}
