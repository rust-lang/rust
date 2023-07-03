//! See [`NavigationTarget`].

use std::fmt;

use either::Either;
use hir::{
    symbols::FileSymbol, AssocItem, Documentation, FieldSource, HasAttrs, HasContainer, HasSource,
    HirDisplay, HirFileId, InFile, LocalSource, ModuleSource,
};
use ide_db::{
    base_db::{FileId, FileRange},
    SymbolKind,
};
use ide_db::{defs::Definition, RootDatabase};
use stdx::never;
use syntax::{
    ast::{self, HasName},
    AstNode, SmolStr, SyntaxNode, TextRange,
};

/// `NavigationTarget` represents an element in the editor's UI which you can
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
    /// A "most interesting" range within the `full_range`.
    ///
    /// Typically, `full_range` is the whole syntax node, including doc
    /// comments, and `focus_range` is the range of the identifier.
    ///
    /// Clients should place the cursor on this range when navigating to this target.
    pub focus_range: Option<TextRange>,
    pub name: SmolStr,
    pub kind: Option<SymbolKind>,
    pub container_name: Option<SmolStr>,
    pub description: Option<String>,
    pub docs: Option<Documentation>,
    /// In addition to a `name` field, a `NavigationTarget` may also be aliased
    /// In such cases we want a `NavigationTarget` to be accessible by its alias
    pub alias: Option<SmolStr>,
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

impl<T: TryToNav, U: TryToNav> TryToNav for Either<T, U> {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        match self {
            Either::Left(it) => it.try_to_nav(db),
            Either::Right(it) => it.try_to_nav(db),
        }
    }
}

impl NavigationTarget {
    pub fn focus_or_full_range(&self) -> TextRange {
        self.focus_range.unwrap_or(self.full_range)
    }

    pub(crate) fn from_module_to_decl(db: &RootDatabase, module: hir::Module) -> NavigationTarget {
        let name = module.name(db).map(|it| it.to_smol_str()).unwrap_or_default();
        if let Some(InFile { value, file_id }) = &module.declaration_source(db) {
            let (file_id, full_range, focus_range) =
                orig_range_with_focus(db, *file_id, value.syntax(), value.name());
            let mut res = NavigationTarget::from_syntax(
                file_id,
                name,
                focus_range,
                full_range,
                SymbolKind::Module,
            );
            res.docs = module.attrs(db).docs();
            res.description = Some(module.display(db).to_string());
            return res;
        }
        module.to_nav(db)
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
            buf.push_str(&format!(" {focus_range:?}"))
        }
        if let Some(container_name) = &self.container_name {
            buf.push_str(&format!(" {container_name}"))
        }
        buf
    }

    /// Allows `NavigationTarget` to be created from a `NameOwner`
    pub(crate) fn from_named(
        db: &RootDatabase,
        InFile { file_id, value }: InFile<&dyn ast::HasName>,
        kind: SymbolKind,
    ) -> NavigationTarget {
        let name = value.name().map(|it| it.text().into()).unwrap_or_else(|| "_".into());

        let (file_id, full_range, focus_range) =
            orig_range_with_focus(db, file_id, value.syntax(), value.name());

        NavigationTarget::from_syntax(file_id, name, focus_range, full_range, kind)
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
            alias: None,
        }
    }
}

impl TryToNav for FileSymbol {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        let full_range = self.loc.original_range(db);
        let focus_range = self.loc.original_name_range(db).and_then(|it| {
            if it.file_id == full_range.file_id {
                Some(it.range)
            } else {
                None
            }
        });

        Some(NavigationTarget {
            file_id: full_range.file_id,
            name: if self.is_alias { self.def.name(db)?.to_smol_str() } else { self.name.clone() },
            alias: if self.is_alias { Some(self.name.clone()) } else { None },
            kind: Some(hir::ModuleDefId::from(self.def).into()),
            full_range: full_range.range,
            focus_range,
            container_name: self.container_name.clone(),
            description: match self.def {
                hir::ModuleDef::Module(it) => Some(it.display(db).to_string()),
                hir::ModuleDef::Function(it) => Some(it.display(db).to_string()),
                hir::ModuleDef::Adt(it) => Some(it.display(db).to_string()),
                hir::ModuleDef::Variant(it) => Some(it.display(db).to_string()),
                hir::ModuleDef::Const(it) => Some(it.display(db).to_string()),
                hir::ModuleDef::Static(it) => Some(it.display(db).to_string()),
                hir::ModuleDef::Trait(it) => Some(it.display(db).to_string()),
                hir::ModuleDef::TraitAlias(it) => Some(it.display(db).to_string()),
                hir::ModuleDef::TypeAlias(it) => Some(it.display(db).to_string()),
                hir::ModuleDef::Macro(it) => Some(it.display(db).to_string()),
                hir::ModuleDef::BuiltinType(_) => None,
            },
            docs: None,
        })
    }
}

impl TryToNav for Definition {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        match self {
            Definition::Local(it) => Some(it.to_nav(db)),
            Definition::Label(it) => Some(it.to_nav(db)),
            Definition::Module(it) => Some(it.to_nav(db)),
            Definition::Macro(it) => it.try_to_nav(db),
            Definition::Field(it) => it.try_to_nav(db),
            Definition::SelfType(it) => it.try_to_nav(db),
            Definition::GenericParam(it) => it.try_to_nav(db),
            Definition::Function(it) => it.try_to_nav(db),
            Definition::Adt(it) => it.try_to_nav(db),
            Definition::Variant(it) => it.try_to_nav(db),
            Definition::Const(it) => it.try_to_nav(db),
            Definition::Static(it) => it.try_to_nav(db),
            Definition::Trait(it) => it.try_to_nav(db),
            Definition::TraitAlias(it) => it.try_to_nav(db),
            Definition::TypeAlias(it) => it.try_to_nav(db),
            Definition::BuiltinType(_) => None,
            Definition::ToolModule(_) => None,
            Definition::BuiltinAttr(_) => None,
            // FIXME: The focus range should be set to the helper declaration
            Definition::DeriveHelper(it) => it.derive().try_to_nav(db),
        }
    }
}

impl TryToNav for hir::ModuleDef {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        match self {
            hir::ModuleDef::Module(it) => Some(it.to_nav(db)),
            hir::ModuleDef::Function(it) => it.try_to_nav(db),
            hir::ModuleDef::Adt(it) => it.try_to_nav(db),
            hir::ModuleDef::Variant(it) => it.try_to_nav(db),
            hir::ModuleDef::Const(it) => it.try_to_nav(db),
            hir::ModuleDef::Static(it) => it.try_to_nav(db),
            hir::ModuleDef::Trait(it) => it.try_to_nav(db),
            hir::ModuleDef::TraitAlias(it) => it.try_to_nav(db),
            hir::ModuleDef::TypeAlias(it) => it.try_to_nav(db),
            hir::ModuleDef::Macro(it) => it.try_to_nav(db),
            hir::ModuleDef::BuiltinType(_) => None,
        }
    }
}

pub(crate) trait ToNavFromAst: Sized {
    const KIND: SymbolKind;
    fn container_name(self, db: &RootDatabase) -> Option<SmolStr> {
        _ = db;
        None
    }
}

fn container_name(db: &RootDatabase, t: impl HasContainer) -> Option<SmolStr> {
    match t.container(db) {
        hir::ItemContainer::Trait(it) => Some(it.name(db).to_smol_str()),
        // FIXME: Handle owners of blocks correctly here
        hir::ItemContainer::Module(it) => it.name(db).map(|name| name.to_smol_str()),
        _ => None,
    }
}

impl ToNavFromAst for hir::Function {
    const KIND: SymbolKind = SymbolKind::Function;
    fn container_name(self, db: &RootDatabase) -> Option<SmolStr> {
        container_name(db, self)
    }
}

impl ToNavFromAst for hir::Const {
    const KIND: SymbolKind = SymbolKind::Const;
    fn container_name(self, db: &RootDatabase) -> Option<SmolStr> {
        container_name(db, self)
    }
}
impl ToNavFromAst for hir::Static {
    const KIND: SymbolKind = SymbolKind::Static;
    fn container_name(self, db: &RootDatabase) -> Option<SmolStr> {
        container_name(db, self)
    }
}
impl ToNavFromAst for hir::Struct {
    const KIND: SymbolKind = SymbolKind::Struct;
    fn container_name(self, db: &RootDatabase) -> Option<SmolStr> {
        container_name(db, self)
    }
}
impl ToNavFromAst for hir::Enum {
    const KIND: SymbolKind = SymbolKind::Enum;
    fn container_name(self, db: &RootDatabase) -> Option<SmolStr> {
        container_name(db, self)
    }
}
impl ToNavFromAst for hir::Variant {
    const KIND: SymbolKind = SymbolKind::Variant;
}
impl ToNavFromAst for hir::Union {
    const KIND: SymbolKind = SymbolKind::Union;
    fn container_name(self, db: &RootDatabase) -> Option<SmolStr> {
        container_name(db, self)
    }
}
impl ToNavFromAst for hir::TypeAlias {
    const KIND: SymbolKind = SymbolKind::TypeAlias;
    fn container_name(self, db: &RootDatabase) -> Option<SmolStr> {
        container_name(db, self)
    }
}
impl ToNavFromAst for hir::Trait {
    const KIND: SymbolKind = SymbolKind::Trait;
    fn container_name(self, db: &RootDatabase) -> Option<SmolStr> {
        container_name(db, self)
    }
}
impl ToNavFromAst for hir::TraitAlias {
    const KIND: SymbolKind = SymbolKind::TraitAlias;
    fn container_name(self, db: &RootDatabase) -> Option<SmolStr> {
        container_name(db, self)
    }
}

impl<D> TryToNav for D
where
    D: HasSource + ToNavFromAst + Copy + HasAttrs + HirDisplay,
    D::Ast: ast::HasName,
{
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        let src = self.source(db)?;
        let mut res = NavigationTarget::from_named(
            db,
            src.as_ref().map(|it| it as &dyn ast::HasName),
            D::KIND,
        );
        res.docs = self.docs(db);
        res.description = Some(self.display(db).to_string());
        res.container_name = self.container_name(db);
        Some(res)
    }
}

impl ToNav for hir::Module {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let InFile { file_id, value } = self.definition_source(db);

        let name = self.name(db).map(|it| it.to_smol_str()).unwrap_or_default();
        let (syntax, focus) = match &value {
            ModuleSource::SourceFile(node) => (node.syntax(), None),
            ModuleSource::Module(node) => (node.syntax(), node.name()),
            ModuleSource::BlockExpr(node) => (node.syntax(), None),
        };
        let (file_id, full_range, focus_range) = orig_range_with_focus(db, file_id, syntax, focus);
        NavigationTarget::from_syntax(file_id, name, focus_range, full_range, SymbolKind::Module)
    }
}

impl TryToNav for hir::Impl {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        let InFile { file_id, value } = self.source(db)?;
        let derive_attr = self.as_builtin_derive(db);

        let (focus, syntax) = match &derive_attr {
            Some(attr) => (None, attr.value.syntax()),
            None => (value.self_ty(), value.syntax()),
        };

        let (file_id, full_range, focus_range) = orig_range_with_focus(db, file_id, syntax, focus);
        Some(NavigationTarget::from_syntax(
            file_id,
            "impl".into(),
            focus_range,
            full_range,
            SymbolKind::Impl,
        ))
    }
}

impl TryToNav for hir::Field {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        let src = self.source(db)?;

        let field_source = match &src.value {
            FieldSource::Named(it) => {
                let mut res =
                    NavigationTarget::from_named(db, src.with_value(it), SymbolKind::Field);
                res.docs = self.docs(db);
                res.description = Some(self.display(db).to_string());
                res
            }
            FieldSource::Pos(it) => {
                let FileRange { file_id, range } =
                    src.with_value(it.syntax()).original_file_range(db);
                NavigationTarget::from_syntax(file_id, "".into(), None, range, SymbolKind::Field)
            }
        };
        Some(field_source)
    }
}

impl TryToNav for hir::Macro {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        let src = self.source(db)?;
        let name_owner: &dyn ast::HasName = match &src.value {
            Either::Left(it) => it,
            Either::Right(it) => it,
        };
        let mut res = NavigationTarget::from_named(
            db,
            src.as_ref().with_value(name_owner),
            self.kind(db).into(),
        );
        res.docs = self.docs(db);
        Some(res)
    }
}

impl TryToNav for hir::Adt {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        match self {
            hir::Adt::Struct(it) => it.try_to_nav(db),
            hir::Adt::Union(it) => it.try_to_nav(db),
            hir::Adt::Enum(it) => it.try_to_nav(db),
        }
    }
}

impl TryToNav for hir::AssocItem {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        match self {
            AssocItem::Function(it) => it.try_to_nav(db),
            AssocItem::Const(it) => it.try_to_nav(db),
            AssocItem::TypeAlias(it) => it.try_to_nav(db),
        }
    }
}

impl TryToNav for hir::GenericParam {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        match self {
            hir::GenericParam::TypeParam(it) => it.try_to_nav(db),
            hir::GenericParam::ConstParam(it) => it.try_to_nav(db),
            hir::GenericParam::LifetimeParam(it) => it.try_to_nav(db),
        }
    }
}

impl ToNav for LocalSource {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let InFile { file_id, value } = &self.source;
        let file_id = *file_id;
        let local = self.local;
        let (node, name) = match &value {
            Either::Left(bind_pat) => (bind_pat.syntax(), bind_pat.name()),
            Either::Right(it) => (it.syntax(), it.name()),
        };

        let (file_id, full_range, focus_range) = orig_range_with_focus(db, file_id, node, name);

        let name = local.name(db).to_smol_str();
        let kind = if local.is_self(db) {
            SymbolKind::SelfParam
        } else if local.is_param(db) {
            SymbolKind::ValueParam
        } else {
            SymbolKind::Local
        };
        NavigationTarget {
            file_id,
            name,
            alias: None,
            kind: Some(kind),
            full_range,
            focus_range,
            container_name: None,
            description: None,
            docs: None,
        }
    }
}

impl ToNav for hir::Local {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        self.primary_source(db).to_nav(db)
    }
}

impl ToNav for hir::Label {
    fn to_nav(&self, db: &RootDatabase) -> NavigationTarget {
        let InFile { file_id, value } = self.source(db);
        let name = self.name(db).to_smol_str();

        let (file_id, full_range, focus_range) =
            orig_range_with_focus(db, file_id, value.syntax(), value.lifetime());

        NavigationTarget {
            file_id,
            name,
            alias: None,
            kind: Some(SymbolKind::Label),
            full_range,
            focus_range,
            container_name: None,
            description: None,
            docs: None,
        }
    }
}

impl TryToNav for hir::TypeParam {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        let InFile { file_id, value } = self.merge().source(db)?;
        let name = self.name(db).to_smol_str();

        let value = match value {
            Either::Left(ast::TypeOrConstParam::Type(x)) => Either::Left(x),
            Either::Left(ast::TypeOrConstParam::Const(_)) => {
                never!();
                return None;
            }
            Either::Right(x) => Either::Right(x),
        };

        let syntax = match &value {
            Either::Left(type_param) => type_param.syntax(),
            Either::Right(trait_) => trait_.syntax(),
        };
        let focus = value.as_ref().either(|it| it.name(), |it| it.name());

        let (file_id, full_range, focus_range) = orig_range_with_focus(db, file_id, syntax, focus);

        Some(NavigationTarget {
            file_id,
            name,
            alias: None,
            kind: Some(SymbolKind::TypeParam),
            full_range,
            focus_range,
            container_name: None,
            description: None,
            docs: None,
        })
    }
}

impl TryToNav for hir::TypeOrConstParam {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        self.split(db).try_to_nav(db)
    }
}

impl TryToNav for hir::LifetimeParam {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        let InFile { file_id, value } = self.source(db)?;
        let name = self.name(db).to_smol_str();

        let FileRange { file_id, range } =
            InFile::new(file_id, value.syntax()).original_file_range(db);
        Some(NavigationTarget {
            file_id,
            name,
            alias: None,
            kind: Some(SymbolKind::LifetimeParam),
            full_range: range,
            focus_range: Some(range),
            container_name: None,
            description: None,
            docs: None,
        })
    }
}

impl TryToNav for hir::ConstParam {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<NavigationTarget> {
        let InFile { file_id, value } = self.merge().source(db)?;
        let name = self.name(db).to_smol_str();

        let value = match value {
            Either::Left(ast::TypeOrConstParam::Const(x)) => x,
            _ => {
                never!();
                return None;
            }
        };

        let (file_id, full_range, focus_range) =
            orig_range_with_focus(db, file_id, value.syntax(), value.name());
        Some(NavigationTarget {
            file_id,
            name,
            alias: None,
            kind: Some(SymbolKind::ConstParam),
            full_range,
            focus_range,
            container_name: None,
            description: None,
            docs: None,
        })
    }
}

fn orig_range_with_focus(
    db: &RootDatabase,
    hir_file: HirFileId,
    value: &SyntaxNode,
    name: Option<impl AstNode>,
) -> (FileId, TextRange, Option<TextRange>) {
    let FileRange { file_id, range: full_range } =
        InFile::new(hir_file, value).original_file_range(db);
    let focus_range = name
        .and_then(|it| InFile::new(hir_file, it.syntax()).original_file_range_opt(db))
        .and_then(|range| if range.file_id == file_id { Some(range.range) } else { None });

    (file_id, full_range, focus_range)
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
