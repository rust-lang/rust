//! This module generates [moniker](https://microsoft.github.io/language-server-protocol/specifications/lsif/0.6.0/specification/#exportsImports)
//! for LSIF and LSP.

use core::fmt;

use hir::{Adt, AsAssocItem, Crate, HirDisplay, MacroKind, Semantics};
use ide_db::{
    FilePosition, RootDatabase,
    base_db::{CrateOrigin, LangCrateOrigin},
    defs::{Definition, IdentClass},
    helpers::pick_best_token,
};
use itertools::Itertools;
use syntax::{AstNode, SyntaxKind::*, T};

use crate::{RangeInfo, doc_links::token_as_doc_comment, parent_module::crates_for};

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
            SymbolInformationKind::AssociatedType => Self::Type,
            SymbolInformationKind::Attribute => Self::Meta,
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
            SymbolInformationKind::StaticVariable => Self::Term,
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
pub enum MonikerResult {
    /// Uniquely identifies a definition.
    Moniker(Moniker),
    /// Specifies that the definition is a local, and so does not have a unique identifier. Provides
    /// a unique identifier for the container.
    Local { enclosing_moniker: Option<Moniker> },
}

impl MonikerResult {
    pub fn from_def(db: &RootDatabase, def: Definition, from_crate: Crate) -> Option<Self> {
        def_to_moniker(db, def, from_crate)
    }
}

/// Information which uniquely identifies a definition which might be referenceable outside of the
/// source file. Visibility declarations do not affect presence.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Moniker {
    pub identifier: MonikerIdentifier,
    pub kind: MonikerKind,
    pub package_information: PackageInformation,
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
            MacroKind::Derive
            | MacroKind::DeriveBuiltIn
            | MacroKind::AttrBuiltIn
            | MacroKind::Attr => Attribute,
            MacroKind::Declarative | MacroKind::DeclarativeBuiltIn | MacroKind::ProcMacro => Macro,
        },
        Definition::Field(..) | Definition::TupleField(..) => Field,
        Definition::Module(..) | Definition::Crate(..) => Module,
        Definition::Function(it) => {
            if it.as_assoc_item(db).is_some() {
                if it.has_self_param(db) {
                    if it.has_body(db) { Method } else { TraitMethod }
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

/// Computes a `MonikerResult` for a definition. Result cases:
///
/// * `Some(MonikerResult::Moniker(_))` provides a unique `Moniker` which refers to a definition.
///
/// * `Some(MonikerResult::Local { .. })` provides a `Moniker` for the definition enclosing a local.
///
/// * `None` is returned for definitions which are not in a module: `BuiltinAttr`, `BuiltinType`,
///   `BuiltinLifetime`, `TupleField`, `ToolModule`, and `InlineAsmRegOrRegClass`. TODO: it might be
///   sensible to provide monikers that refer to some non-existent crate of compiler builtin
///   definitions.
pub(crate) fn def_to_moniker(
    db: &RootDatabase,
    definition: Definition,
    from_crate: Crate,
) -> Option<MonikerResult> {
    match definition {
        Definition::Local(_) | Definition::Label(_) | Definition::GenericParam(_) => {
            return Some(MonikerResult::Local {
                enclosing_moniker: enclosing_def_to_moniker(db, definition, from_crate),
            });
        }
        _ => {}
    }
    Some(MonikerResult::Moniker(def_to_non_local_moniker(db, definition, from_crate)?))
}

fn enclosing_def_to_moniker(
    db: &RootDatabase,
    mut def: Definition,
    from_crate: Crate,
) -> Option<Moniker> {
    loop {
        let enclosing_def = def.enclosing_definition(db)?;
        if let Some(enclosing_moniker) = def_to_non_local_moniker(db, enclosing_def, from_crate) {
            return Some(enclosing_moniker);
        }
        def = enclosing_def;
    }
}

fn def_to_non_local_moniker(
    db: &RootDatabase,
    definition: Definition,
    from_crate: Crate,
) -> Option<Moniker> {
    let module = match definition {
        Definition::Module(module) if module.is_crate_root() => module,
        _ => definition.module(db)?,
    };
    let krate = module.krate();
    let edition = krate.edition(db);

    // Add descriptors for this definition and every enclosing definition.
    let mut reverse_description = vec![];
    let mut def = definition;
    loop {
        match def {
            Definition::SelfType(impl_) => {
                if let Some(trait_ref) = impl_.trait_ref(db) {
                    // Trait impls use the trait type for the 2nd parameter.
                    reverse_description.push(MonikerDescriptor {
                        name: display(db, module, trait_ref),
                        desc: MonikerDescriptorKind::TypeParameter,
                    });
                }
                // Both inherent and trait impls use the self type for the first parameter.
                reverse_description.push(MonikerDescriptor {
                    name: display(db, module, impl_.self_ty(db)),
                    desc: MonikerDescriptorKind::TypeParameter,
                });
                reverse_description.push(MonikerDescriptor {
                    name: "impl".to_owned(),
                    desc: MonikerDescriptorKind::Type,
                });
            }
            _ => {
                if let Some(name) = def.name(db) {
                    reverse_description.push(MonikerDescriptor {
                        name: name.display(db, edition).to_string(),
                        desc: def_to_kind(db, def).into(),
                    });
                } else {
                    match def {
                        Definition::Module(module) if module.is_crate_root() => {
                            // only include `crate` namespace by itself because we prefer
                            // `rust-analyzer cargo foo . bar/` over `rust-analyzer cargo foo . crate/bar/`
                            if reverse_description.is_empty() {
                                reverse_description.push(MonikerDescriptor {
                                    name: "crate".to_owned(),
                                    desc: MonikerDescriptorKind::Namespace,
                                });
                            }
                        }
                        _ => {
                            tracing::error!(?def, "Encountered enclosing definition with no name");
                        }
                    }
                }
            }
        }
        let Some(next_def) = def.enclosing_definition(db) else {
            break;
        };
        def = next_def;
    }
    if reverse_description.is_empty() {
        return None;
    }
    reverse_description.reverse();
    let description = reverse_description;

    Some(Moniker {
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

fn display<T: HirDisplay>(db: &RootDatabase, module: hir::Module, it: T) -> String {
    match it.display_source_code(db, module.into(), true) {
        Ok(result) => result,
        // Fallback on display variant that always succeeds
        Err(_) => {
            let fallback_result = it.display(db, module.krate().to_display_target(db)).to_string();
            tracing::error!(
                display = %fallback_result, "`display_source_code` failed; falling back to using display"
            );
            fallback_result
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{MonikerResult, fixture};

    use super::MonikerKind;

    #[allow(dead_code)]
    #[track_caller]
    fn no_moniker(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        let (analysis, position) = fixture::position(ra_fixture);
        if let Some(x) = analysis.moniker(position).unwrap() {
            assert_eq!(x.info.len(), 0, "Moniker found but no moniker expected: {x:?}");
        }
    }

    #[track_caller]
    fn check_local_moniker(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        identifier: &str,
        package: &str,
        kind: MonikerKind,
    ) {
        let (analysis, position) = fixture::position(ra_fixture);
        let x = analysis.moniker(position).unwrap().expect("no moniker found").info;
        assert_eq!(x.len(), 1);
        match x.into_iter().next().unwrap() {
            MonikerResult::Local { enclosing_moniker: Some(x) } => {
                assert_eq!(identifier, x.identifier.to_string());
                assert_eq!(package, format!("{:?}", x.package_information));
                assert_eq!(kind, x.kind);
            }
            MonikerResult::Local { enclosing_moniker: None } => {
                panic!("Unexpected local with no enclosing moniker");
            }
            MonikerResult::Moniker(_) => {
                panic!("Unexpected non-local moniker");
            }
        }
    }

    #[track_caller]
    fn check_moniker(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        identifier: &str,
        package: &str,
        kind: MonikerKind,
    ) {
        let (analysis, position) = fixture::position(ra_fixture);
        let x = analysis.moniker(position).unwrap().expect("no moniker found").info;
        assert_eq!(x.len(), 1);
        match x.into_iter().next().unwrap() {
            MonikerResult::Local { enclosing_moniker } => {
                panic!("Unexpected local enclosed in {enclosing_moniker:?}");
            }
            MonikerResult::Moniker(x) => {
                assert_eq!(identifier, x.identifier.to_string());
                assert_eq!(package, format!("{:?}", x.package_information));
                assert_eq!(kind, x.kind);
            }
        }
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
            "foo::module::impl::MyStruct::MyTrait::func",
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
    fn local() {
        check_local_moniker(
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
            "foo::module::func",
            r#"PackageInformation { name: "foo", repo: Some("https://a.b/foo.git"), version: Some("0.1.0") }"#,
            MonikerKind::Export,
        );
    }
}
