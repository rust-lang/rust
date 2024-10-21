use rustc_hash::FxHashMap;
use rustdoc_json_types::{FORMAT_VERSION, Item, ItemKind, Visibility};

use super::*;
use crate::json_find::SelectorPart;

#[track_caller]
fn check(krate: &Crate, errs: &[Error]) {
    let krate_string = serde_json::to_string(krate).unwrap();
    let krate_json = serde_json::from_str(&krate_string).unwrap();

    let mut validator = Validator::new(krate, krate_json);
    validator.check_crate();

    assert_eq!(errs, &validator.errs[..]);
}

#[test]
fn errors_on_missing_links() {
    let k = Crate {
        root: Id(0),
        crate_version: None,
        includes_private: false,
        index: FxHashMap::from_iter([(Id(0), Item {
            name: Some("root".to_owned()),
            id: Id(0),
            crate_id: 0,
            span: None,
            visibility: Visibility::Public,
            docs: None,
            links: FxHashMap::from_iter([("Not Found".to_owned(), Id(1))]),
            attrs: vec![],
            deprecation: None,
            inner: ItemEnum::Module(Module { is_crate: true, items: vec![], is_stripped: false }),
        })]),
        paths: FxHashMap::default(),
        external_crates: FxHashMap::default(),
        format_version: rustdoc_json_types::FORMAT_VERSION,
    };

    check(&k, &[Error {
        kind: ErrorKind::NotFound(vec![vec![
            SelectorPart::Field("index".to_owned()),
            SelectorPart::Field("0".to_owned()),
            SelectorPart::Field("links".to_owned()),
            SelectorPart::Field("Not Found".to_owned()),
        ]]),
        id: Id(1),
    }]);
}

// Test we would catch
// https://github.com/rust-lang/rust/issues/104064#issuecomment-1368589718
#[test]
fn errors_on_local_in_paths_and_not_index() {
    let krate = Crate {
        root: Id(0),
        crate_version: None,
        includes_private: false,
        index: FxHashMap::from_iter([
            (Id(0), Item {
                id: Id(0),
                crate_id: 0,
                name: Some("microcore".to_owned()),
                span: None,
                visibility: Visibility::Public,
                docs: None,
                links: FxHashMap::from_iter([(("prim@i32".to_owned(), Id(2)))]),
                attrs: Vec::new(),
                deprecation: None,
                inner: ItemEnum::Module(Module {
                    is_crate: true,
                    items: vec![Id(1)],
                    is_stripped: false,
                }),
            }),
            (Id(1), Item {
                id: Id(1),
                crate_id: 0,
                name: Some("i32".to_owned()),
                span: None,
                visibility: Visibility::Public,
                docs: None,
                links: FxHashMap::default(),
                attrs: Vec::new(),
                deprecation: None,
                inner: ItemEnum::Primitive(Primitive { name: "i32".to_owned(), impls: vec![] }),
            }),
        ]),
        paths: FxHashMap::from_iter([(Id(2), ItemSummary {
            crate_id: 0,
            path: vec!["microcore".to_owned(), "i32".to_owned()],
            kind: ItemKind::Primitive,
        })]),
        external_crates: FxHashMap::default(),
        format_version: rustdoc_json_types::FORMAT_VERSION,
    };

    check(&krate, &[Error {
        id: Id(2),
        kind: ErrorKind::Custom("Id for local item in `paths` but not in `index`".to_owned()),
    }]);
}

#[test]
#[should_panic = "LOCAL_CRATE_ID is wrong"]
fn checks_local_crate_id_is_correct() {
    let krate = Crate {
        root: Id(0),
        crate_version: None,
        includes_private: false,
        index: FxHashMap::from_iter([(Id(0), Item {
            id: Id(0),
            crate_id: LOCAL_CRATE_ID.wrapping_add(1),
            name: Some("irrelavent".to_owned()),
            span: None,
            visibility: Visibility::Public,
            docs: None,
            links: FxHashMap::default(),
            attrs: Vec::new(),
            deprecation: None,
            inner: ItemEnum::Module(Module { is_crate: true, items: vec![], is_stripped: false }),
        })]),
        paths: FxHashMap::default(),
        external_crates: FxHashMap::default(),
        format_version: FORMAT_VERSION,
    };
    check(&krate, &[]);
}
