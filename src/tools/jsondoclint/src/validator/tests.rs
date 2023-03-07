use rustc_hash::FxHashMap;
use rustdoc_json_types::{Crate, Item, ItemKind, ItemSummary, Visibility, FORMAT_VERSION};

use crate::json_find::SelectorPart;

use super::*;

#[track_caller]
fn check(krate: &Crate, errs: &[Error]) {
    let krate_string = serde_json::to_string(krate).unwrap();
    let krate_json = serde_json::from_str(&krate_string).unwrap();

    let mut validator = Validator::new(krate, krate_json);
    validator.check_crate();

    assert_eq!(errs, &validator.errs[..]);
}

fn id(s: &str) -> Id {
    Id(s.to_owned())
}

#[test]
fn errors_on_missing_links() {
    let k = Crate {
        root: id("0"),
        crate_version: None,
        includes_private: false,
        index: FxHashMap::from_iter([(
            id("0"),
            Item {
                name: Some("root".to_owned()),
                id: id(""),
                crate_id: 0,
                span: None,
                visibility: Visibility::Public,
                docs: None,
                links: FxHashMap::from_iter([("Not Found".to_owned(), id("1"))]),
                attrs: vec![],
                deprecation: None,
                inner: ItemEnum::Module(Module {
                    is_crate: true,
                    items: vec![],
                    is_stripped: false,
                }),
            },
        )]),
        paths: FxHashMap::default(),
        external_crates: FxHashMap::default(),
        format_version: rustdoc_json_types::FORMAT_VERSION,
    };

    check(
        &k,
        &[Error {
            kind: ErrorKind::NotFound(vec![vec![
                SelectorPart::Field("index".to_owned()),
                SelectorPart::Field("0".to_owned()),
                SelectorPart::Field("links".to_owned()),
                SelectorPart::Field("Not Found".to_owned()),
            ]]),
            id: id("1"),
        }],
    );
}

// Test we would catch
// https://github.com/rust-lang/rust/issues/104064#issuecomment-1368589718
#[test]
fn errors_on_local_in_paths_and_not_index() {
    let krate = Crate {
        root: id("0:0:1572"),
        crate_version: None,
        includes_private: false,
        index: FxHashMap::from_iter([
            (
                id("0:0:1572"),
                Item {
                    id: id("0:0:1572"),
                    crate_id: 0,
                    name: Some("microcore".to_owned()),
                    span: None,
                    visibility: Visibility::Public,
                    docs: None,
                    links: FxHashMap::from_iter([(("prim@i32".to_owned(), id("0:1:1571")))]),
                    attrs: Vec::new(),
                    deprecation: None,
                    inner: ItemEnum::Module(Module {
                        is_crate: true,
                        items: vec![id("0:1:717")],
                        is_stripped: false,
                    }),
                },
            ),
            (
                id("0:1:717"),
                Item {
                    id: id("0:1:717"),
                    crate_id: 0,
                    name: Some("i32".to_owned()),
                    span: None,
                    visibility: Visibility::Public,
                    docs: None,
                    links: FxHashMap::default(),
                    attrs: Vec::new(),
                    deprecation: None,
                    inner: ItemEnum::Primitive(Primitive { name: "i32".to_owned(), impls: vec![] }),
                },
            ),
        ]),
        paths: FxHashMap::from_iter([(
            id("0:1:1571"),
            ItemSummary {
                crate_id: 0,
                path: vec!["microcore".to_owned(), "i32".to_owned()],
                kind: ItemKind::Primitive,
            },
        )]),
        external_crates: FxHashMap::default(),
        format_version: rustdoc_json_types::FORMAT_VERSION,
    };

    check(
        &krate,
        &[Error {
            id: id("0:1:1571"),
            kind: ErrorKind::Custom("Id for local item in `paths` but not in `index`".to_owned()),
        }],
    );
}

#[test]
#[should_panic = "LOCAL_CRATE_ID is wrong"]
fn checks_local_crate_id_is_correct() {
    let krate = Crate {
        root: id("root"),
        crate_version: None,
        includes_private: false,
        index: FxHashMap::from_iter([(
            id("root"),
            Item {
                id: id("root"),
                crate_id: LOCAL_CRATE_ID.wrapping_add(1),
                name: Some("irrelavent".to_owned()),
                span: None,
                visibility: Visibility::Public,
                docs: None,
                links: FxHashMap::default(),
                attrs: Vec::new(),
                deprecation: None,
                inner: ItemEnum::Module(Module {
                    is_crate: true,
                    items: vec![],
                    is_stripped: false,
                }),
            },
        )]),
        paths: FxHashMap::default(),
        external_crates: FxHashMap::default(),
        format_version: FORMAT_VERSION,
    };
    check(&krate, &[]);
}
