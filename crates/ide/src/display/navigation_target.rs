//! FIXME: write short doc here

use std::fmt;

use either::Either;
use hir::{AssocItem, Documentation, FieldSource, HasAttrs, HasSource, InFile, ModuleSource};
use ide_db::{
    base_db::{FileId, SourceDatabase},
    symbol_index::FileSymbolKind,
};
use ide_db::{defs::Definition, RootDatabase};
use syntax::{
    ast::{self, NameOwner},
    match_ast, AstNode, SmolStr, TextRange,
};

use crate::FileSymbol;

use super::short_label::ShortLabel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SymbolKind {
    Module,
    Impl,
    Field,
    TypeParam,
    LifetimeParam,
    ValueParam,
    SelfParam,
    Local,
    Function,
    Const,
    Static,
    Struct,
    Enum,
    Variant,
    Union,
    TypeAlias,
    Trait,
    Macro,
}

/// `NavigationTarget` represents and element in the editor's UI which you can
/// click on to navigate to a particular piece of code.
///
/// Typically, a `NavigationTarget` corresponds to some element in the source
/// code, like a function or a struct, but this is not strictly required.
#[derive(Clone, PartialEq, Eq, Hash)]
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
    pub kind: Option<SymbolKind>,
    pub container_name: Option<SmolStr>,
    pub description: Option<String>,
    pub docs: Option<Documentation>,
}

impl fmt::Debug for NavigationTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut f = f.debug_struct("NavigationTarget");
        macro_rules! opt {
            ($($name:ident)*) => {$(
                if let Some(it) = &self.$name {
                    f.field(stringify!($name), it);
                }
            )*}
        }
        f.field("file_id", &self.file_id).field("full_range", &self.full_range);
        opt!(focus_range);
        f.field("name", &self.name);
        opt!(kind container_name description docs);
        f.finish()
    }
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
            let node = src.as_ref().map(|it| it.syntax());
            let frange = node.original_file_range(db);
            let mut res = NavigationTarget::from_syntax(
                frange.file_id,
                name,
                None,
                frange.range,
                SymbolKind::Module,
            );
            res.docs = module.attrs(db).docs();
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
        let mut buf = format!(
            "{} {:?} {:?} {:?}",
            self.name,
            self.kind.unwrap(),
            self.file_id,
            self.full_range
        );
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
        kind: SymbolKind,
    ) -> NavigationTarget {
        let name =
            node.value.name().map(|it| it.text().clone()).unwrap_or_else(|| SmolStr::new("_"));
        let focus_range =
            node.value.name().map(|it| node.with_value(it.syntax()).original_file_range(db).range);
        let frange = node.map(|it| it.syntax()).original_file_range(db);

        NavigationTarget::from_syntax(frange.file_id, name, focus_range, frange.range, kind)
    }

    fn from_syntax(
        file_id: FileId,
        name: SmolStr,
        focus_range: Option<TextRange>,
        full_range: TextRange,
        kind: SymbolKind,
    ) -> NavigationTarget {
        NavigationTarget {
            file_id,
            name,
            kind: Some(kind),
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
            kind: Some(match self.kind {
                FileSymbolKind::Function => SymbolKind::Function,
                FileSymbolKind::Struct => SymbolKind::Struct,
                FileSymbolKind::Enum => SymbolKind::Enum,
                FileSymbolKind::Trait => SymbolKind::Trait,
                FileSymbolKind::Module => SymbolKind::Module,
                FileSymbolKind::TypeAlias => SymbolKind::TypeAlias,
                FileSymbolKind::Const => SymbolKind::Const,
                FileSymbolKind::Static => SymbolKind::Static,
                FileSymbolKind::Macro => SymbolKind::Macro,
            }),
            full_range: self.range,
            focus_range: self.name_range,
            container_name: self.container_name.clone(),
            description: description_from_symbol(db, self),
            docs: None,
        }
    }
}

impl TryToNav for Definition {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        match self {
            Definition::Macro(it) => {
                // FIXME: Currently proc-macro do not have ast-node,
                // such that it does not have source
                // more discussion: https://github.com/rust-analyzer/rust-analyzer/issues/6913
                if it.is_proc_macro() {
                    return None;
                }
                Some(it.to_nav(db))
            }
            Definition::Field(it) => Some(it.to_nav(db)),
            Definition::ModuleDef(it) => it.try_to_nav(db),
            Definition::SelfType(it) => Some(it.to_nav(db)),
            Definition::Local(it) => Some(it.to_nav(db)),
            Definition::TypeParam(it) => Some(it.to_nav(db)),
            Definition::LifetimeParam(it) => Some(it.to_nav(db)),
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

pub(crate) trait ToNavFromAst {
    const KIND: SymbolKind;
}
impl ToNavFromAst for hir::Function {
    const KIND: SymbolKind = SymbolKind::Function;
}
impl ToNavFromAst for hir::Const {
    const KIND: SymbolKind = SymbolKind::Const;
}
impl ToNavFromAst for hir::Static {
    const KIND: SymbolKind = SymbolKind::Static;
}
impl ToNavFromAst for hir::Struct {
    const KIND: SymbolKind = SymbolKind::Struct;
}
impl ToNavFromAst for hir::Enum {
    const KIND: SymbolKind = SymbolKind::Enum;
}
impl ToNavFromAst for hir::EnumVariant {
    const KIND: SymbolKind = SymbolKind::Variant;
}
impl ToNavFromAst for hir::Union {
    const KIND: SymbolKind = SymbolKind::Union;
}
impl ToNavFromAst for hir::TypeAlias {
    const KIND: SymbolKind = SymbolKind::TypeAlias;
}
impl ToNavFromAst for hir::Trait {
    const KIND: SymbolKind = SymbolKind::Trait;
}

impl<D> ToNav for D
where
    D: HasSource + ToNavFromAst + Copy + HasAttrs,
    D::Ast: ast::NameOwner + ShortLabel,
{
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let src = self.source(db);
        let mut res = NavigationTarget::from_named(
            db,
            src.as_ref().map(|it| it as &dyn ast::NameOwner),
            D::KIND,
        );
        res.docs = self.docs(db);
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
        let frange = src.with_value(syntax).original_file_range(db);
        NavigationTarget::from_syntax(frange.file_id, name, focus, frange.range, SymbolKind::Module)
    }
}

impl ToNav for hir::Impl {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let src = self.source(db);
        let derive_attr = self.is_builtin_derive(db);
        let frange = if let Some(item) = &derive_attr {
            item.syntax().original_file_range(db)
        } else {
            src.as_ref().map(|it| it.syntax()).original_file_range(db)
        };
        let focus_range = if derive_attr.is_some() {
            None
        } else {
            src.value.self_ty().map(|ty| src.with_value(ty.syntax()).original_file_range(db).range)
        };

        NavigationTarget::from_syntax(
            frange.file_id,
            "impl".into(),
            focus_range,
            frange.range,
            SymbolKind::Impl,
        )
    }
}

impl ToNav for hir::Field {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let src = self.source(db);

        match &src.value {
            FieldSource::Named(it) => {
                let mut res =
                    NavigationTarget::from_named(db, src.with_value(it), SymbolKind::Field);
                res.docs = self.docs(db);
                res.description = it.short_label();
                res
            }
            FieldSource::Pos(it) => {
                let frange = src.with_value(it.syntax()).original_file_range(db);
                NavigationTarget::from_syntax(
                    frange.file_id,
                    "".into(),
                    None,
                    frange.range,
                    SymbolKind::Field,
                )
            }
        }
    }
}

impl ToNav for hir::MacroDef {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let src = self.source(db);
        log::debug!("nav target {:#?}", src.value.syntax());
        let mut res = NavigationTarget::from_named(
            db,
            src.as_ref().map(|it| it as &dyn ast::NameOwner),
            SymbolKind::Macro,
        );
        res.docs = self.docs(db);
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
        let full_range = src.with_value(&node).original_file_range(db);
        let name = match self.name(db) {
            Some(it) => it.to_string().into(),
            None => "".into(),
        };
        let kind = if self.is_param(db) { SymbolKind::ValueParam } else { SymbolKind::Local };
        NavigationTarget {
            file_id: full_range.file_id,
            name,
            kind: Some(kind),
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
            kind: Some(SymbolKind::TypeParam),
            full_range,
            focus_range,
            container_name: None,
            description: None,
            docs: None,
        }
    }
}

impl ToNav for hir::LifetimeParam {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let src = self.source(db);
        let full_range = src.value.syntax().text_range();
        NavigationTarget {
            file_id: src.file_id.original_file(db),
            name: self.name(db).to_string().into(),
            kind: Some(SymbolKind::LifetimeParam),
            full_range,
            focus_range: Some(full_range),
            container_name: None,
            description: None,
            docs: None,
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

    use crate::{fixture, Query};

    #[test]
    fn test_nav_for_symbol() {
        let (analysis, _) = fixture::file(
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
                        0,
                    ),
                    full_range: 0..17,
                    focus_range: 5..13,
                    name: "FooInner",
                    kind: Enum,
                    description: "enum FooInner",
                },
                NavigationTarget {
                    file_id: FileId(
                        0,
                    ),
                    full_range: 29..46,
                    focus_range: 34..42,
                    name: "FooInner",
                    kind: Enum,
                    container_name: "foo",
                    description: "enum FooInner",
                },
            ]
        "#]]
        .assert_debug_eq(&navs);
    }

    #[test]
    fn test_world_symbols_are_case_sensitive() {
        let (analysis, _) = fixture::file(
            r#"
fn foo() {}
struct Foo;
"#,
        );

        let navs = analysis.symbol_search(Query::new("foo".to_string())).unwrap();
        assert_eq!(navs.len(), 2)
    }
}
