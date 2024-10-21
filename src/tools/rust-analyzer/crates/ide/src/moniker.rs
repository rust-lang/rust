//! This module generates [moniker](https://microsoft.github.io/language-server-protocol/specifications/lsif/0.6.0/specification/#exportsImports)
//! for LSIF and LSP.

use core::fmt;

use hir::{Adt, AsAssocItem, AssocItemContainer, Crate, MacroKind, Semantics};
use ide_db::{
    base_db::{CrateOrigin, LangCrateOrigin},
    defs::{Definition, IdentClass},
    helpers::pick_best_token,
    FilePosition, RootDatabase,
};
use itertools::Itertools;
use syntax::{AstNode, SyntaxKind::*, T};

use crate::{doc_links::token_as_doc_comment, parent_module::crates_for, RangeInfo};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MonikerDescriptorKind {
    Namespace,
    Type,
    Term,
    Method,
    TypeParameter,
    Parameter,
    Macro,
    Meta,
}

// Subset of scip_types::SymbolInformation::Kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SymbolInformationKind {
    AssociatedType,
    Attribute,
    Constant,
    Enum,
    EnumMember,
    Field,
    Function,
    Macro,
    Method,
    Module,
    Parameter,
    SelfParameter,
    StaticMethod,
    StaticVariable,
    Struct,
    Trait,
    TraitMethod,
    Type,
    TypeAlias,
    TypeParameter,
    Union,
    Variable,
}

impl From<SymbolInformationKind> for MonikerDescriptorKind {
    fn from(value: SymbolInformationKind) -> Self {
        match value {
            SymbolInformationKind::AssociatedType => Self::TypeParameter,
            SymbolInformationKind::Attribute => Self::Macro,
            SymbolInformationKind::Constant => Self::Term,
            SymbolInformationKind::Enum => Self::Type,
            SymbolInformationKind::EnumMember => Self::Type,
            SymbolInformationKind::Field => Self::Term,
            SymbolInformationKind::Function => Self::Method,
            SymbolInformationKind::Macro => Self::Macro,
            SymbolInformationKind::Method => Self::Method,
            SymbolInformationKind::Module => Self::Namespace,
            SymbolInformationKind::Parameter => Self::Parameter,
            SymbolInformationKind::SelfParameter => Self::Parameter,
            SymbolInformationKind::StaticMethod => Self::Method,
            SymbolInformationKind::StaticVariable => Self::Meta,
            SymbolInformationKind::Struct => Self::Type,
            SymbolInformationKind::Trait => Self::Type,
            SymbolInformationKind::TraitMethod => Self::Method,
            SymbolInformationKind::Type => Self::Type,
            SymbolInformationKind::TypeAlias => Self::Type,
            SymbolInformationKind::TypeParameter => Self::TypeParameter,
            SymbolInformationKind::Union => Self::Type,
            SymbolInformationKind::Variable => Self::Term,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonikerDescriptor {
    pub name: String,
    pub desc: MonikerDescriptorKind,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonikerIdentifier {
    pub crate_name: String,
    pub description: Vec<MonikerDescriptor>,
}

impl fmt::Display for MonikerIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.crate_name)?;
        f.write_fmt(format_args!("::{}", self.description.iter().map(|x| &x.name).join("::")))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MonikerKind {
    Import,
    Export,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MonikerResult {
    pub identifier: MonikerIdentifier,
    pub kind: MonikerKind,
    pub package_information: PackageInformation,
}

impl MonikerResult {
    pub fn from_def(db: &RootDatabase, def: Definition, from_crate: Crate) -> Option<Self> {
        def_to_moniker(db, def, from_crate)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackageInformation {
    pub name: String,
    pub repo: Option<String>,
    pub version: Option<String>,
}

pub(crate) fn moniker(
    db: &RootDatabase,
    FilePosition { file_id, offset }: FilePosition,
) -> Option<RangeInfo<Vec<MonikerResult>>> {
    let sema = &Semantics::new(db);
    let file = sema.parse_guess_edition(file_id).syntax().clone();
    let current_crate: hir::Crate = crates_for(db, file_id).pop()?.into();
    let original_token = pick_best_token(file.token_at_offset(offset), |kind| match kind {
        IDENT
        | INT_NUMBER
        | LIFETIME_IDENT
        | T![self]
        | T![super]
        | T![crate]
        | T![Self]
        | COMMENT => 2,
        kind if kind.is_trivia() => 0,
        _ => 1,
    })?;
    if let Some(doc_comment) = token_as_doc_comment(&original_token) {
        return doc_comment.get_definition_with_descend_at(sema, offset, |def, _, _| {
            let m = def_to_moniker(db, def, current_crate)?;
            Some(RangeInfo::new(original_token.text_range(), vec![m]))
        });
    }
    let navs = sema
        .descend_into_macros_exact(original_token.clone())
        .into_iter()
        .filter_map(|token| {
            IdentClass::classify_token(sema, &token).map(IdentClass::definitions_no_ops).map(|it| {
                it.into_iter().flat_map(|def| def_to_moniker(sema.db, def, current_crate))
            })
        })
        .flatten()
        .unique()
        .collect::<Vec<_>>();
    Some(RangeInfo::new(original_token.text_range(), navs))
}

pub(crate) fn def_to_kind(db: &RootDatabase, def: Definition) -> SymbolInformationKind {
    use SymbolInformationKind::*;

    match def {
        Definition::Macro(it) => match it.kind(db) {
            MacroKind::Declarative => Macro,
            MacroKind::Derive => Attribute,
            MacroKind::BuiltIn => Macro,
            MacroKind::Attr => Attribute,
            MacroKind::ProcMacro => Macro,
        },
        Definition::Field(..) | Definition::TupleField(..) => Field,
        Definition::Module(..) => Module,
        Definition::Function(it) => {
            if it.as_assoc_item(db).is_some() {
                if it.has_self_param(db) {
                    if it.has_body(db) {
                        Method
                    } else {
                        TraitMethod
                    }
                } else {
                    StaticMethod
                }
            } else {
                Function
            }
        }
        Definition::Adt(Adt::Struct(..)) => Struct,
        Definition::Adt(Adt::Union(..)) => Union,
        Definition::Adt(Adt::Enum(..)) => Enum,
        Definition::Variant(..) => EnumMember,
        Definition::Const(..) => Constant,
        Definition::Static(..) => StaticVariable,
        Definition::Trait(..) => Trait,
        Definition::TraitAlias(..) => Trait,
        Definition::TypeAlias(it) => {
            if it.as_assoc_item(db).is_some() {
                AssociatedType
            } else {
                TypeAlias
            }
        }
        Definition::BuiltinType(..) => Type,
        Definition::BuiltinLifetime(_) => TypeParameter,
        Definition::SelfType(..) => TypeAlias,
        Definition::GenericParam(..) => TypeParameter,
        Definition::Local(it) => {
            if it.is_self(db) {
                SelfParameter
            } else if it.is_param(db) {
                Parameter
            } else {
                Variable
            }
        }
        Definition::Label(..) | Definition::InlineAsmOperand(_) => Variable, // For lack of a better variant
        Definition::DeriveHelper(..) => Attribute,
        Definition::BuiltinAttr(..) => Attribute,
        Definition::ToolModule(..) => Module,
        Definition::ExternCrateDecl(..) => Module,
        Definition::InlineAsmRegOrRegClass(..) => Module,
    }
}

pub(crate) fn def_to_moniker(
    db: &RootDatabase,
    def: Definition,
    from_crate: Crate,
) -> Option<MonikerResult> {
    if matches!(
        def,
        Definition::GenericParam(_)
            | Definition::Label(_)
            | Definition::DeriveHelper(_)
            | Definition::BuiltinAttr(_)
            | Definition::ToolModule(_)
    ) {
        return None;
    }

    let module = def.module(db)?;
    let krate = module.krate();
    let edition = krate.edition(db);
    let mut description = vec![];
    description.extend(module.path_to_root(db).into_iter().filter_map(|x| {
        Some(MonikerDescriptor {
            name: x.name(db)?.display(db, edition).to_string(),
            desc: def_to_kind(db, x.into()).into(),
        })
    }));

    // Handle associated items within a trait
    if let Some(assoc) = def.as_assoc_item(db) {
        let container = assoc.container(db);
        match container {
            AssocItemContainer::Trait(trait_) => {
                // Because different traits can have functions with the same name,
                // we have to include the trait name as part of the moniker for uniqueness.
                description.push(MonikerDescriptor {
                    name: trait_.name(db).display(db, edition).to_string(),
                    desc: def_to_kind(db, trait_.into()).into(),
                });
            }
            AssocItemContainer::Impl(impl_) => {
                // Because a struct can implement multiple traits, for implementations
                // we add both the struct name and the trait name to the path
                if let Some(adt) = impl_.self_ty(db).as_adt() {
                    description.push(MonikerDescriptor {
                        name: adt.name(db).display(db, edition).to_string(),
                        desc: def_to_kind(db, adt.into()).into(),
                    });
                }

                if let Some(trait_) = impl_.trait_(db) {
                    description.push(MonikerDescriptor {
                        name: trait_.name(db).display(db, edition).to_string(),
                        desc: def_to_kind(db, trait_.into()).into(),
                    });
                }
            }
        }
    }

    if let Definition::Field(it) = def {
        description.push(MonikerDescriptor {
            name: it.parent_def(db).name(db).display(db, edition).to_string(),
            desc: def_to_kind(db, it.parent_def(db).into()).into(),
        });
    }

    // Qualify locals/parameters by their parent definition name.
    if let Definition::Local(it) = def {
        let parent = Definition::try_from(it.parent(db)).ok();
        if let Some(parent) = parent {
            let parent_name = parent.name(db);
            if let Some(name) = parent_name {
                description.push(MonikerDescriptor {
                    name: name.display(db, edition).to_string(),
                    desc: def_to_kind(db, parent).into(),
                });
            }
        }
    }

    let desc = def_to_kind(db, def).into();

    let name_desc = match def {
        // These are handled by top-level guard (for performance).
        Definition::GenericParam(_)
        | Definition::Label(_)
        | Definition::DeriveHelper(_)
        | Definition::BuiltinLifetime(_)
        | Definition::BuiltinAttr(_)
        | Definition::ToolModule(_)
        | Definition::InlineAsmRegOrRegClass(_)
        | Definition::InlineAsmOperand(_) => return None,

        Definition::Local(local) => {
            if !local.is_param(db) {
                return None;
            }

            MonikerDescriptor { name: local.name(db).display(db, edition).to_string(), desc }
        }
        Definition::Macro(m) => {
            MonikerDescriptor { name: m.name(db).display(db, edition).to_string(), desc }
        }
        Definition::Function(f) => {
            MonikerDescriptor { name: f.name(db).display(db, edition).to_string(), desc }
        }
        Definition::Variant(v) => {
            MonikerDescriptor { name: v.name(db).display(db, edition).to_string(), desc }
        }
        Definition::Const(c) => {
            MonikerDescriptor { name: c.name(db)?.display(db, edition).to_string(), desc }
        }
        Definition::Trait(trait_) => {
            MonikerDescriptor { name: trait_.name(db).display(db, edition).to_string(), desc }
        }
        Definition::TraitAlias(ta) => {
            MonikerDescriptor { name: ta.name(db).display(db, edition).to_string(), desc }
        }
        Definition::TypeAlias(ta) => {
            MonikerDescriptor { name: ta.name(db).display(db, edition).to_string(), desc }
        }
        Definition::Module(m) => {
            MonikerDescriptor { name: m.name(db)?.display(db, edition).to_string(), desc }
        }
        Definition::BuiltinType(b) => {
            MonikerDescriptor { name: b.name().display(db, edition).to_string(), desc }
        }
        Definition::SelfType(imp) => MonikerDescriptor {
            name: imp.self_ty(db).as_adt()?.name(db).display(db, edition).to_string(),
            desc,
        },
        Definition::Field(it) => {
            MonikerDescriptor { name: it.name(db).display(db, edition).to_string(), desc }
        }
        Definition::TupleField(it) => {
            MonikerDescriptor { name: it.name().display(db, edition).to_string(), desc }
        }
        Definition::Adt(adt) => {
            MonikerDescriptor { name: adt.name(db).display(db, edition).to_string(), desc }
        }
        Definition::Static(s) => {
            MonikerDescriptor { name: s.name(db).display(db, edition).to_string(), desc }
        }
        Definition::ExternCrateDecl(m) => {
            MonikerDescriptor { name: m.name(db).display(db, edition).to_string(), desc }
        }
    };

    description.push(name_desc);

    Some(MonikerResult {
        identifier: MonikerIdentifier {
            crate_name: krate.display_name(db)?.crate_name().to_string(),
            description,
        },
        kind: if krate == from_crate { MonikerKind::Export } else { MonikerKind::Import },
        package_information: {
            let (name, repo, version) = match krate.origin(db) {
                CrateOrigin::Library { repo, name } => (name, repo, krate.version(db)),
                CrateOrigin::Local { repo, name } => (
                    name.unwrap_or(krate.display_name(db)?.canonical_name().to_owned()),
                    repo,
                    krate.version(db),
                ),
                CrateOrigin::Rustc { name } => (
                    name.clone(),
                    Some("https://github.com/rust-lang/rust/".to_owned()),
                    Some(format!("https://github.com/rust-lang/rust/compiler/{name}",)),
                ),
                CrateOrigin::Lang(lang) => (
                    krate.display_name(db)?.canonical_name().to_owned(),
                    Some("https://github.com/rust-lang/rust/".to_owned()),
                    Some(match lang {
                        LangCrateOrigin::Other => {
                            "https://github.com/rust-lang/rust/library/".into()
                        }
                        lang => format!("https://github.com/rust-lang/rust/library/{lang}",),
                    }),
                ),
            };
            PackageInformation { name: name.as_str().to_owned(), repo, version }
        },
    })
}

#[cfg(test)]
mod tests {
    use crate::fixture;

    use super::MonikerKind;

    #[track_caller]
    fn no_moniker(ra_fixture: &str) {
        let (analysis, position) = fixture::position(ra_fixture);
        if let Some(x) = analysis.moniker(position).unwrap() {
            assert_eq!(x.info.len(), 0, "Moniker founded but no moniker expected: {x:?}");
        }
    }

    #[track_caller]
    fn check_moniker(ra_fixture: &str, identifier: &str, package: &str, kind: MonikerKind) {
        let (analysis, position) = fixture::position(ra_fixture);
        let x = analysis.moniker(position).unwrap().expect("no moniker found").info;
        assert_eq!(x.len(), 1);
        let x = x.into_iter().next().unwrap();
        assert_eq!(identifier, x.identifier.to_string());
        assert_eq!(package, format!("{:?}", x.package_information));
        assert_eq!(kind, x.kind);
    }

    #[test]
    fn basic() {
        check_moniker(
            r#"
//- /lib.rs crate:main deps:foo
use foo::module::func;
fn main() {
    func$0();
}
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod module {
    pub fn func() {}
}
"#,
            "foo::module::func",
            r#"PackageInformation { name: "foo", repo: Some("https://a.b/foo.git"), version: Some("0.1.0") }"#,
            MonikerKind::Import,
        );
        check_moniker(
            r#"
//- /lib.rs crate:main deps:foo
use foo::module::func;
fn main() {
    func();
}
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod module {
    pub fn func$0() {}
}
"#,
            "foo::module::func",
            r#"PackageInformation { name: "foo", repo: Some("https://a.b/foo.git"), version: Some("0.1.0") }"#,
            MonikerKind::Export,
        );
    }

    #[test]
    fn moniker_for_trait() {
        check_moniker(
            r#"
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod module {
    pub trait MyTrait {
        pub fn func$0() {}
    }
}
"#,
            "foo::module::MyTrait::func",
            r#"PackageInformation { name: "foo", repo: Some("https://a.b/foo.git"), version: Some("0.1.0") }"#,
            MonikerKind::Export,
        );
    }

    #[test]
    fn moniker_for_trait_constant() {
        check_moniker(
            r#"
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod module {
    pub trait MyTrait {
        const MY_CONST$0: u8;
    }
}
"#,
            "foo::module::MyTrait::MY_CONST",
            r#"PackageInformation { name: "foo", repo: Some("https://a.b/foo.git"), version: Some("0.1.0") }"#,
            MonikerKind::Export,
        );
    }

    #[test]
    fn moniker_for_trait_type() {
        check_moniker(
            r#"
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod module {
    pub trait MyTrait {
        type MyType$0;
    }
}
"#,
            "foo::module::MyTrait::MyType",
            r#"PackageInformation { name: "foo", repo: Some("https://a.b/foo.git"), version: Some("0.1.0") }"#,
            MonikerKind::Export,
        );
    }

    #[test]
    fn moniker_for_trait_impl_function() {
        check_moniker(
            r#"
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod module {
    pub trait MyTrait {
        pub fn func() {}
    }

    struct MyStruct {}

    impl MyTrait for MyStruct {
        pub fn func$0() {}
    }
}
"#,
            "foo::module::MyStruct::MyTrait::func",
            r#"PackageInformation { name: "foo", repo: Some("https://a.b/foo.git"), version: Some("0.1.0") }"#,
            MonikerKind::Export,
        );
    }

    #[test]
    fn moniker_for_field() {
        check_moniker(
            r#"
//- /lib.rs crate:main deps:foo
use foo::St;
fn main() {
    let x = St { a$0: 2 };
}
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub struct St {
    pub a: i32,
}
"#,
            "foo::St::a",
            r#"PackageInformation { name: "foo", repo: Some("https://a.b/foo.git"), version: Some("0.1.0") }"#,
            MonikerKind::Import,
        );
    }

    #[test]
    fn no_moniker_for_local() {
        no_moniker(
            r#"
//- /lib.rs crate:main deps:foo
use foo::module::func;
fn main() {
    func();
}
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod module {
    pub fn func() {
        let x$0 = 2;
    }
}
"#,
        );
    }
}
