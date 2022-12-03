use std::collections::HashMap;

use rustdoc_json_types::{Crate, Item, Visibility};

use super::*;

#[track_caller]
fn check(krate: &Crate, errs: &[Error]) {
    let mut validator = Validator::new(krate);
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
        index: HashMap::from_iter([(
            id("0"),
            Item {
                name: Some("root".to_owned()),
                id: id(""),
                crate_id: 0,
                span: None,
                visibility: Visibility::Public,
                docs: None,
                links: HashMap::from_iter([("Not Found".to_owned(), id("1"))]),
                attrs: vec![],
                deprecation: None,
                inner: ItemEnum::Module(Module {
                    is_crate: true,
                    items: vec![],
                    is_stripped: false,
                }),
            },
        )]),
        paths: HashMap::new(),
        external_crates: HashMap::new(),
        format_version: rustdoc_json_types::FORMAT_VERSION,
    };

    check(&k, &[Error { kind: ErrorKind::NotFound, id: id("1") }]);
}
