//! See [`NavigationTarget`].

use std::fmt;

use arrayvec::ArrayVec;
use either::Either;
use hir::{
    db::ExpandDatabase, symbols::FileSymbol, AssocItem, FieldSource, HasContainer, HasSource,
    HirDisplay, HirFileId, InFile, LocalSource, ModuleSource,
};
use ide_db::{
    base_db::{FileId, FileRange},
    defs::Definition,
    documentation::{Documentation, HasDocs},
    RootDatabase, SymbolKind,
};
use stdx::never;
use syntax::{
    ast::{self, HasName},
    format_smolstr, AstNode, SmolStr, SyntaxNode, TextRange,
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
    ///
    /// This range must be contained within [`Self::full_range`].
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
    fn to_nav(&self, db: &RootDatabase) -> UpmappingResult<NavigationTarget>;
}

pub trait TryToNav {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>>;
}

impl<T: TryToNav, U: TryToNav> TryToNav for Either<T, U> {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
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

    pub(crate) fn from_module_to_decl(
        db: &RootDatabase,
        module: hir::Module,
    ) -> UpmappingResult<NavigationTarget> {
        let name = module.name(db).map(|it| it.to_smol_str()).unwrap_or_default();
        match module.declaration_source(db) {
            Some(InFile { value, file_id }) => {
                orig_range_with_focus(db, file_id, value.syntax(), value.name()).map(
                    |(FileRange { file_id, range: full_range }, focus_range)| {
                        let mut res = NavigationTarget::from_syntax(
                            file_id,
                            name.clone(),
                            focus_range,
                            full_range,
                            SymbolKind::Module,
                        );
                        res.docs = module.docs(db);
                        res.description = Some(module.display(db).to_string());
                        res
                    },
                )
            }
            _ => module.to_nav(db),
        }
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
    ) -> UpmappingResult<NavigationTarget> {
        let name: SmolStr = value.name().map(|it| it.text().into()).unwrap_or_else(|| "_".into());

        orig_range_with_focus(db, file_id, value.syntax(), value.name()).map(
            |(FileRange { file_id, range: full_range }, focus_range)| {
                NavigationTarget::from_syntax(file_id, name.clone(), focus_range, full_range, kind)
            },
        )
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
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
        Some(
            orig_range_with_focus_r(
                db,
                self.loc.hir_file_id,
                self.loc.ptr.text_range(),
                Some(self.loc.name_ptr.text_range()),
            )
            .map(|(FileRange { file_id, range: full_range }, focus_range)| {
                NavigationTarget {
                    file_id,
                    name: self
                        .is_alias
                        .then(|| self.def.name(db))
                        .flatten()
                        .map_or_else(|| self.name.clone(), |it| it.to_smol_str()),
                    alias: self.is_alias.then(|| self.name.clone()),
                    kind: Some(hir::ModuleDefId::from(self.def).into()),
                    full_range,
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
                }
            }),
        )
    }
}

impl TryToNav for Definition {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
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
            Definition::ExternCrateDecl(it) => Some(it.try_to_nav(db)?),
            Definition::BuiltinType(_) | Definition::TupleField(_) => None,
            Definition::ToolModule(_) => None,
            Definition::BuiltinAttr(_) => None,
            // FIXME: The focus range should be set to the helper declaration
            Definition::DeriveHelper(it) => it.derive().try_to_nav(db),
        }
    }
}

impl TryToNav for hir::ModuleDef {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
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
    D: HasSource + ToNavFromAst + Copy + HasDocs + HirDisplay,
    D::Ast: ast::HasName,
{
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
        let src = self.source(db)?;
        Some(
            NavigationTarget::from_named(
                db,
                src.as_ref().map(|it| it as &dyn ast::HasName),
                D::KIND,
            )
            .map(|mut res| {
                res.docs = self.docs(db);
                res.description = Some(self.display(db).to_string());
                res.container_name = self.container_name(db);
                res
            }),
        )
    }
}

impl ToNav for hir::Module {
    fn to_nav(&self, db: &RootDatabase) -> UpmappingResult<NavigationTarget> {
        let InFile { file_id, value } = self.definition_source(db);

        let name = self.name(db).map(|it| it.to_smol_str()).unwrap_or_default();
        let (syntax, focus) = match &value {
            ModuleSource::SourceFile(node) => (node.syntax(), None),
            ModuleSource::Module(node) => (node.syntax(), node.name()),
            ModuleSource::BlockExpr(node) => (node.syntax(), None),
        };

        orig_range_with_focus(db, file_id, syntax, focus).map(
            |(FileRange { file_id, range: full_range }, focus_range)| {
                NavigationTarget::from_syntax(
                    file_id,
                    name.clone(),
                    focus_range,
                    full_range,
                    SymbolKind::Module,
                )
            },
        )
    }
}

impl TryToNav for hir::Impl {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
        let InFile { file_id, value } = self.source(db)?;
        let derive_path = self.as_builtin_derive_path(db);

        let (file_id, focus, syntax) = match &derive_path {
            Some(attr) => (attr.file_id.into(), None, attr.value.syntax()),
            None => (file_id, value.self_ty(), value.syntax()),
        };

        Some(orig_range_with_focus(db, file_id, syntax, focus).map(
            |(FileRange { file_id, range: full_range }, focus_range)| {
                NavigationTarget::from_syntax(
                    file_id,
                    "impl".into(),
                    focus_range,
                    full_range,
                    SymbolKind::Impl,
                )
            },
        ))
    }
}

impl TryToNav for hir::ExternCrateDecl {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
        let src = self.source(db)?;
        let InFile { file_id, value } = src;
        let focus = value
            .rename()
            .map_or_else(|| value.name_ref().map(Either::Left), |it| it.name().map(Either::Right));

        Some(orig_range_with_focus(db, file_id, value.syntax(), focus).map(
            |(FileRange { file_id, range: full_range }, focus_range)| {
                let mut res = NavigationTarget::from_syntax(
                    file_id,
                    self.alias_or_name(db).unwrap_or_else(|| self.name(db)).to_smol_str(),
                    focus_range,
                    full_range,
                    SymbolKind::Module,
                );

                res.docs = self.docs(db);
                res.description = Some(self.display(db).to_string());
                res.container_name = container_name(db, *self);
                res
            },
        ))
    }
}

impl TryToNav for hir::Field {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
        let src = self.source(db)?;

        let field_source = match &src.value {
            FieldSource::Named(it) => {
                NavigationTarget::from_named(db, src.with_value(it), SymbolKind::Field).map(
                    |mut res| {
                        res.docs = self.docs(db);
                        res.description = Some(self.display(db).to_string());
                        res
                    },
                )
            }
            FieldSource::Pos(it) => orig_range(db, src.file_id, it.syntax()).map(
                |(FileRange { file_id, range: full_range }, focus_range)| {
                    NavigationTarget::from_syntax(
                        file_id,
                        format_smolstr!("{}", self.index()),
                        focus_range,
                        full_range,
                        SymbolKind::Field,
                    )
                },
            ),
        };
        Some(field_source)
    }
}

impl TryToNav for hir::Macro {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
        let src = self.source(db)?;
        let name_owner: &dyn ast::HasName = match &src.value {
            Either::Left(it) => it,
            Either::Right(it) => it,
        };
        Some(
            NavigationTarget::from_named(
                db,
                src.as_ref().with_value(name_owner),
                self.kind(db).into(),
            )
            .map(|mut res| {
                res.docs = self.docs(db);
                res
            }),
        )
    }
}

impl TryToNav for hir::Adt {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
        match self {
            hir::Adt::Struct(it) => it.try_to_nav(db),
            hir::Adt::Union(it) => it.try_to_nav(db),
            hir::Adt::Enum(it) => it.try_to_nav(db),
        }
    }
}

impl TryToNav for hir::AssocItem {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
        match self {
            AssocItem::Function(it) => it.try_to_nav(db),
            AssocItem::Const(it) => it.try_to_nav(db),
            AssocItem::TypeAlias(it) => it.try_to_nav(db),
        }
    }
}

impl TryToNav for hir::GenericParam {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
        match self {
            hir::GenericParam::TypeParam(it) => it.try_to_nav(db),
            hir::GenericParam::ConstParam(it) => it.try_to_nav(db),
            hir::GenericParam::LifetimeParam(it) => it.try_to_nav(db),
        }
    }
}

impl ToNav for LocalSource {
    fn to_nav(&self, db: &RootDatabase) -> UpmappingResult<NavigationTarget> {
        let InFile { file_id, value } = &self.source;
        let file_id = *file_id;
        let local = self.local;
        let (node, name) = match &value {
            Either::Left(bind_pat) => (bind_pat.syntax(), bind_pat.name()),
            Either::Right(it) => (it.syntax(), it.name()),
        };

        orig_range_with_focus(db, file_id, node, name).map(
            |(FileRange { file_id, range: full_range }, focus_range)| {
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
            },
        )
    }
}

impl ToNav for hir::Local {
    fn to_nav(&self, db: &RootDatabase) -> UpmappingResult<NavigationTarget> {
        self.primary_source(db).to_nav(db)
    }
}

impl ToNav for hir::Label {
    fn to_nav(&self, db: &RootDatabase) -> UpmappingResult<NavigationTarget> {
        let InFile { file_id, value } = self.source(db);
        let name = self.name(db).to_smol_str();

        orig_range_with_focus(db, file_id, value.syntax(), value.lifetime()).map(
            |(FileRange { file_id, range: full_range }, focus_range)| NavigationTarget {
                file_id,
                name: name.clone(),
                alias: None,
                kind: Some(SymbolKind::Label),
                full_range,
                focus_range,
                container_name: None,
                description: None,
                docs: None,
            },
        )
    }
}

impl TryToNav for hir::TypeParam {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
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

        Some(orig_range_with_focus(db, file_id, syntax, focus).map(
            |(FileRange { file_id, range: full_range }, focus_range)| NavigationTarget {
                file_id,
                name: name.clone(),
                alias: None,
                kind: Some(SymbolKind::TypeParam),
                full_range,
                focus_range,
                container_name: None,
                description: None,
                docs: None,
            },
        ))
    }
}

impl TryToNav for hir::TypeOrConstParam {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
        self.split(db).try_to_nav(db)
    }
}

impl TryToNav for hir::LifetimeParam {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
        let InFile { file_id, value } = self.source(db)?;
        let name = self.name(db).to_smol_str();

        Some(orig_range(db, file_id, value.syntax()).map(
            |(FileRange { file_id, range: full_range }, focus_range)| NavigationTarget {
                file_id,
                name: name.clone(),
                alias: None,
                kind: Some(SymbolKind::LifetimeParam),
                full_range,
                focus_range,
                container_name: None,
                description: None,
                docs: None,
            },
        ))
    }
}

impl TryToNav for hir::ConstParam {
    fn try_to_nav(&self, db: &RootDatabase) -> Option<UpmappingResult<NavigationTarget>> {
        let InFile { file_id, value } = self.merge().source(db)?;
        let name = self.name(db).to_smol_str();

        let value = match value {
            Either::Left(ast::TypeOrConstParam::Const(x)) => x,
            _ => {
                never!();
                return None;
            }
        };

        Some(orig_range_with_focus(db, file_id, value.syntax(), value.name()).map(
            |(FileRange { file_id, range: full_range }, focus_range)| NavigationTarget {
                file_id,
                name: name.clone(),
                alias: None,
                kind: Some(SymbolKind::ConstParam),
                full_range,
                focus_range,
                container_name: None,
                description: None,
                docs: None,
            },
        ))
    }
}

#[derive(Debug)]
pub struct UpmappingResult<T> {
    /// The macro call site.
    pub call_site: T,
    /// The macro definition site, if relevant.
    pub def_site: Option<T>,
}

impl<T> UpmappingResult<T> {
    pub fn call_site(self) -> T {
        self.call_site
    }

    pub fn collect<FI: FromIterator<T>>(self) -> FI {
        FI::from_iter(self)
    }
}

impl<T> IntoIterator for UpmappingResult<T> {
    type Item = T;

    type IntoIter = <ArrayVec<T, 2> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.def_site
            .into_iter()
            .chain(Some(self.call_site))
            .collect::<ArrayVec<_, 2>>()
            .into_iter()
    }
}

impl<T> UpmappingResult<T> {
    fn map<U>(self, f: impl Fn(T) -> U) -> UpmappingResult<U> {
        UpmappingResult { call_site: f(self.call_site), def_site: self.def_site.map(f) }
    }
}

/// Returns the original range of the syntax node, and the range of the name mapped out of macro expansions
/// May return two results if the mapped node originates from a macro definition in which case the
/// second result is the creating macro call.
fn orig_range_with_focus(
    db: &RootDatabase,
    hir_file: HirFileId,
    value: &SyntaxNode,
    name: Option<impl AstNode>,
) -> UpmappingResult<(FileRange, Option<TextRange>)> {
    orig_range_with_focus_r(
        db,
        hir_file,
        value.text_range(),
        name.map(|it| it.syntax().text_range()),
    )
}

fn orig_range_with_focus_r(
    db: &RootDatabase,
    hir_file: HirFileId,
    value: TextRange,
    name: Option<TextRange>,
) -> UpmappingResult<(FileRange, Option<TextRange>)> {
    let Some(name) = name else { return orig_range_r(db, hir_file, value) };

    let call_kind =
        || db.lookup_intern_macro_call(hir_file.macro_file().unwrap().macro_call_id).kind;

    let def_range = || {
        db.lookup_intern_macro_call(hir_file.macro_file().unwrap().macro_call_id)
            .def
            .definition_range(db)
    };

    let value_range = InFile::new(hir_file, value).original_node_file_range_opt(db);
    let ((call_site_range, call_site_focus), def_site) =
        match InFile::new(hir_file, name).original_node_file_range_opt(db) {
            // call site name
            Some((focus_range, ctxt)) if ctxt.is_root() => {
                // Try to upmap the node as well, if it ends up in the def site, go back to the call site
                (
                    (
                        match value_range {
                            // name is in the node in the macro input so we can return it
                            Some((range, ctxt))
                                if ctxt.is_root()
                                    && range.file_id == focus_range.file_id
                                    && range.range.contains_range(focus_range.range) =>
                            {
                                range
                            }
                            // name lies outside the node, so instead point to the macro call which
                            // *should* contain the name
                            _ => {
                                let kind = call_kind();
                                let range = kind.clone().original_call_range_with_body(db);
                                //If the focus range is in the attribute/derive body, we
                                // need to point the call site to the entire body, if not, fall back
                                // to the name range of the attribute/derive call
                                // FIXME: Do this differently, this is very inflexible the caller
                                // should choose this behavior
                                if range.file_id == focus_range.file_id
                                    && range.range.contains_range(focus_range.range)
                                {
                                    range
                                } else {
                                    kind.original_call_range(db)
                                }
                            }
                        },
                        Some(focus_range),
                    ),
                    // no def site relevant
                    None,
                )
            }

            // def site name
            // FIXME: This can be de improved
            Some((focus_range, _ctxt)) => {
                match value_range {
                    // but overall node is in macro input
                    Some((range, ctxt)) if ctxt.is_root() => (
                        // node mapped up in call site, show the node
                        (range, None),
                        // def site, if the name is in the (possibly) upmapped def site range, show the
                        // def site
                        {
                            let (def_site, _) = def_range().original_node_file_range(db);
                            (def_site.file_id == focus_range.file_id
                                && def_site.range.contains_range(focus_range.range))
                            .then_some((def_site, Some(focus_range)))
                        },
                    ),
                    // node is in macro def, just show the focus
                    _ => (
                        // show the macro call
                        (call_kind().original_call_range(db), None),
                        Some((focus_range, Some(focus_range))),
                    ),
                }
            }
            // lost name? can't happen for single tokens
            None => return orig_range_r(db, hir_file, value),
        };

    UpmappingResult {
        call_site: (
            call_site_range,
            call_site_focus.and_then(|FileRange { file_id, range }| {
                if call_site_range.file_id == file_id && call_site_range.range.contains_range(range)
                {
                    Some(range)
                } else {
                    None
                }
            }),
        ),
        def_site: def_site.map(|(def_site_range, def_site_focus)| {
            (
                def_site_range,
                def_site_focus.and_then(|FileRange { file_id, range }| {
                    if def_site_range.file_id == file_id
                        && def_site_range.range.contains_range(range)
                    {
                        Some(range)
                    } else {
                        None
                    }
                }),
            )
        }),
    }
}

fn orig_range(
    db: &RootDatabase,
    hir_file: HirFileId,
    value: &SyntaxNode,
) -> UpmappingResult<(FileRange, Option<TextRange>)> {
    UpmappingResult {
        call_site: (InFile::new(hir_file, value).original_file_range_rooted(db), None),
        def_site: None,
    }
}

fn orig_range_r(
    db: &RootDatabase,
    hir_file: HirFileId,
    value: TextRange,
) -> UpmappingResult<(FileRange, Option<TextRange>)> {
    UpmappingResult {
        call_site: (InFile::new(hir_file, value).original_node_file_range(db).0, None),
        def_site: None,
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

        let navs = analysis.symbol_search(Query::new("FooInner".to_owned()), !0).unwrap();
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

        let navs = analysis.symbol_search(Query::new("foo".to_owned()), !0).unwrap();
        assert_eq!(navs.len(), 2)
    }
}
