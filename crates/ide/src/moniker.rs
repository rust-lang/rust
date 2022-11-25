//! This module generates [moniker](https://microsoft.github.io/language-server-protocol/specifications/lsif/0.6.0/specification/#exportsImports)
//! for LSIF and LSP.

use hir::{db::DefDatabase, AsAssocItem, AssocItemContainer, Crate, Name, Semantics};
use ide_db::{
    base_db::{CrateOrigin, FileId, FileLoader, FilePosition, LangCrateOrigin},
    defs::{Definition, IdentClass},
    helpers::pick_best_token,
    RootDatabase,
};
use itertools::Itertools;
use syntax::{AstNode, SyntaxKind::*, T};

use crate::{doc_links::token_as_doc_comment, RangeInfo};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonikerIdentifier {
    crate_name: String,
    path: Vec<Name>,
}

impl ToString for MonikerIdentifier {
    fn to_string(&self) -> String {
        match self {
            MonikerIdentifier { path, crate_name } => {
                format!("{}::{}", crate_name, path.iter().map(|x| x.to_string()).join("::"))
            }
        }
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackageInformation {
    pub name: String,
    pub repo: String,
    pub version: String,
}

pub(crate) fn crate_for_file(db: &RootDatabase, file_id: FileId) -> Option<Crate> {
    for &krate in db.relevant_crates(file_id).iter() {
        let crate_def_map = db.crate_def_map(krate);
        for (_, data) in crate_def_map.modules() {
            if data.origin.file_id() == Some(file_id) {
                return Some(krate.into());
            }
        }
    }
    None
}

pub(crate) fn moniker(
    db: &RootDatabase,
    FilePosition { file_id, offset }: FilePosition,
) -> Option<RangeInfo<Vec<MonikerResult>>> {
    let sema = &Semantics::new(db);
    let file = sema.parse(file_id).syntax().clone();
    let current_crate = crate_for_file(db, file_id)?;
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
        .descend_into_macros(original_token.clone())
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

pub(crate) fn def_to_moniker(
    db: &RootDatabase,
    def: Definition,
    from_crate: Crate,
) -> Option<MonikerResult> {
    if matches!(def, Definition::GenericParam(_) | Definition::SelfType(_) | Definition::Local(_)) {
        return None;
    }
    let module = def.module(db)?;
    let krate = module.krate();
    let mut path = vec![];
    path.extend(module.path_to_root(db).into_iter().filter_map(|x| x.name(db)));

    // Handle associated items within a trait
    if let Some(assoc) = def.as_assoc_item(db) {
        let container = assoc.container(db);
        match container {
            AssocItemContainer::Trait(trait_) => {
                // Because different traits can have functions with the same name,
                // we have to include the trait name as part of the moniker for uniqueness.
                path.push(trait_.name(db));
            }
            AssocItemContainer::Impl(impl_) => {
                // Because a struct can implement multiple traits, for implementations
                // we add both the struct name and the trait name to the path
                if let Some(adt) = impl_.self_ty(db).as_adt() {
                    path.push(adt.name(db));
                }

                if let Some(trait_) = impl_.trait_(db) {
                    path.push(trait_.name(db));
                }
            }
        }
    }

    if let Definition::Field(it) = def {
        path.push(it.parent_def(db).name(db));
    }

    path.push(def.name(db)?);
    Some(MonikerResult {
        identifier: MonikerIdentifier {
            crate_name: krate.display_name(db)?.crate_name().to_string(),
            path,
        },
        kind: if krate == from_crate { MonikerKind::Export } else { MonikerKind::Import },
        package_information: {
            let name = krate.display_name(db)?.to_string();
            let (repo, version) = match krate.origin(db) {
                CrateOrigin::CratesIo { repo } => (repo?, krate.version(db)?),
                CrateOrigin::Lang(lang) => (
                    "https://github.com/rust-lang/rust/".to_string(),
                    match lang {
                        LangCrateOrigin::Other => {
                            "https://github.com/rust-lang/rust/library/".into()
                        }
                        lang => format!("https://github.com/rust-lang/rust/library/{lang}",),
                    },
                ),
            };
            PackageInformation { name, repo, version }
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
            assert_eq!(x.info.len(), 0, "Moniker founded but no moniker expected: {:?}", x);
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
//- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
pub mod module {
    pub fn func() {}
}
"#,
            "foo::module::func",
            r#"PackageInformation { name: "foo", repo: "https://a.b/foo.git", version: "0.1.0" }"#,
            MonikerKind::Import,
        );
        check_moniker(
            r#"
//- /lib.rs crate:main deps:foo
use foo::module::func;
fn main() {
    func();
}
//- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
pub mod module {
    pub fn func$0() {}
}
"#,
            "foo::module::func",
            r#"PackageInformation { name: "foo", repo: "https://a.b/foo.git", version: "0.1.0" }"#,
            MonikerKind::Export,
        );
    }

    #[test]
    fn moniker_for_trait() {
        check_moniker(
            r#"
//- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
pub mod module {
    pub trait MyTrait {
        pub fn func$0() {}
    }
}
"#,
            "foo::module::MyTrait::func",
            r#"PackageInformation { name: "foo", repo: "https://a.b/foo.git", version: "0.1.0" }"#,
            MonikerKind::Export,
        );
    }

    #[test]
    fn moniker_for_trait_constant() {
        check_moniker(
            r#"
//- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
pub mod module {
    pub trait MyTrait {
        const MY_CONST$0: u8;
    }
}
"#,
            "foo::module::MyTrait::MY_CONST",
            r#"PackageInformation { name: "foo", repo: "https://a.b/foo.git", version: "0.1.0" }"#,
            MonikerKind::Export,
        );
    }

    #[test]
    fn moniker_for_trait_type() {
        check_moniker(
            r#"
//- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
pub mod module {
    pub trait MyTrait {
        type MyType$0;
    }
}
"#,
            "foo::module::MyTrait::MyType",
            r#"PackageInformation { name: "foo", repo: "https://a.b/foo.git", version: "0.1.0" }"#,
            MonikerKind::Export,
        );
    }

    #[test]
    fn moniker_for_trait_impl_function() {
        check_moniker(
            r#"
//- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
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
            r#"PackageInformation { name: "foo", repo: "https://a.b/foo.git", version: "0.1.0" }"#,
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
//- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
pub struct St {
    pub a: i32,
}
"#,
            "foo::St::a",
            r#"PackageInformation { name: "foo", repo: "https://a.b/foo.git", version: "0.1.0" }"#,
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
//- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
pub mod module {
    pub fn func() {
        let x$0 = 2;
    }
}
"#,
        );
    }
}
