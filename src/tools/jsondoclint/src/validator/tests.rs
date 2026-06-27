use rustc_hash::FxHashMap;
use rustdoc_json_types::{
    Abi, FORMAT_VERSION, FunctionHeader, Item, ItemKind, ProvidedDefaultUnstable, Visibility,
};

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
        index: FxHashMap::from_iter([(
            Id(0),
            Item {
                name: Some("root".to_owned()),
                id: Id(0),
                crate_id: 0,
                span: None,
                visibility: Visibility::Public,
                docs: None,
                links: FxHashMap::from_iter([("Not Found".to_owned(), Id(1))]),
                attrs: vec![],
                deprecation: None,
                stability: None,
                const_stability: None,
                inner: ItemEnum::Module(Module {
                    is_crate: true,
                    items: vec![],
                    is_stripped: false,
                }),
            },
        )]),
        paths: FxHashMap::default(),
        external_crates: FxHashMap::default(),
        target: rustdoc_json_types::Target { triple: "".to_string(), target_features: vec![] },
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
            id: Id(1),
        }],
    );
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
            (
                Id(0),
                Item {
                    id: Id(0),
                    crate_id: 0,
                    name: Some("microcore".to_owned()),
                    span: None,
                    visibility: Visibility::Public,
                    docs: None,
                    links: FxHashMap::from_iter([("prim@i32".to_owned(), Id(2))]),
                    attrs: Vec::new(),
                    deprecation: None,
                    stability: None,
                    const_stability: None,
                    inner: ItemEnum::Module(Module {
                        is_crate: true,
                        items: vec![Id(1)],
                        is_stripped: false,
                    }),
                },
            ),
            (
                Id(1),
                Item {
                    id: Id(1),
                    crate_id: 0,
                    name: Some("i32".to_owned()),
                    span: None,
                    visibility: Visibility::Public,
                    docs: None,
                    links: FxHashMap::default(),
                    attrs: Vec::new(),
                    deprecation: None,
                    stability: None,
                    const_stability: None,
                    inner: ItemEnum::Primitive(Primitive { name: "i32".to_owned(), impls: vec![] }),
                },
            ),
        ]),
        paths: FxHashMap::from_iter([(
            Id(2),
            ItemSummary {
                crate_id: 0,
                path: vec!["microcore".to_owned(), "i32".to_owned()],
                kind: ItemKind::Primitive,
            },
        )]),
        external_crates: FxHashMap::default(),
        target: rustdoc_json_types::Target { triple: "".to_string(), target_features: vec![] },
        format_version: rustdoc_json_types::FORMAT_VERSION,
    };

    check(
        &krate,
        &[Error {
            id: Id(2),
            kind: ErrorKind::Custom("Id for local item in `paths` but not in `index`".to_owned()),
        }],
    );
}

#[test]
fn errors_on_missing_path() {
    // crate-name=foo
    // ```
    // pub struct Bar;
    // pub fn mk_bar() -> Bar { ... }
    // ```

    let generics = Generics { params: vec![], where_predicates: vec![] };

    let krate = Crate {
        root: Id(0),
        crate_version: None,
        includes_private: false,
        index: FxHashMap::from_iter([
            (
                Id(0),
                Item {
                    id: Id(0),
                    crate_id: 0,
                    name: Some("foo".to_owned()),
                    span: None,
                    visibility: Visibility::Public,
                    docs: None,
                    links: FxHashMap::default(),
                    attrs: Vec::new(),
                    deprecation: None,
                    stability: None,
                    const_stability: None,
                    inner: ItemEnum::Module(Module {
                        is_crate: true,
                        items: vec![Id(1), Id(2)],
                        is_stripped: false,
                    }),
                },
            ),
            (
                Id(1),
                Item {
                    id: Id(0),
                    crate_id: 0,
                    name: Some("Bar".to_owned()),
                    span: None,
                    visibility: Visibility::Public,
                    docs: None,
                    links: FxHashMap::default(),
                    attrs: Vec::new(),
                    deprecation: None,
                    stability: None,
                    const_stability: None,
                    inner: ItemEnum::Struct(Struct {
                        kind: StructKind::Unit,
                        generics: generics.clone(),
                        impls: vec![],
                    }),
                },
            ),
            (
                Id(2),
                Item {
                    id: Id(0),
                    crate_id: 0,
                    name: Some("mk_bar".to_owned()),
                    span: None,
                    visibility: Visibility::Public,
                    docs: None,
                    links: FxHashMap::default(),
                    attrs: Vec::new(),
                    deprecation: None,
                    stability: None,
                    const_stability: None,
                    inner: ItemEnum::Function(Function {
                        sig: FunctionSignature {
                            inputs: vec![],
                            output: Some(Type::ResolvedPath(Path {
                                path: "Bar".to_owned(),
                                id: Id(1),
                                args: None,
                            })),
                            is_c_variadic: false,
                        },
                        generics,
                        header: FunctionHeader {
                            is_const: false,
                            is_unsafe: false,
                            is_async: false,
                            abi: Abi::Rust,
                        },
                        has_body: true,
                        default_unstable: None,
                    }),
                },
            ),
        ]),
        paths: FxHashMap::from_iter([(
            Id(0),
            ItemSummary { crate_id: 0, path: vec!["foo".to_owned()], kind: ItemKind::Module },
        )]),
        external_crates: FxHashMap::default(),
        target: rustdoc_json_types::Target { triple: "".to_string(), target_features: vec![] },
        format_version: rustdoc_json_types::FORMAT_VERSION,
    };

    check(
        &krate,
        &[Error {
            kind: ErrorKind::Custom(
                r#"No entry in '$.paths' for Path { path: "Bar", id: Id(1), args: None }"#
                    .to_owned(),
            ),
            id: Id(1),
        }],
    );
}

fn krate_with_trait_item(inner: ItemEnum) -> Crate {
    let item_id = Id(2);
    Crate {
        root: Id(0),
        crate_version: None,
        includes_private: false,
        index: FxHashMap::from_iter([
            (
                Id(0),
                Item {
                    id: Id(0),
                    crate_id: 0,
                    name: Some("root".to_owned()),
                    span: None,
                    visibility: Visibility::Public,
                    docs: None,
                    links: FxHashMap::default(),
                    attrs: Vec::new(),
                    deprecation: None,
                    stability: None,
                    const_stability: None,
                    inner: ItemEnum::Module(Module {
                        is_crate: true,
                        items: vec![Id(1)],
                        is_stripped: false,
                    }),
                },
            ),
            (
                Id(1),
                Item {
                    id: Id(1),
                    crate_id: 0,
                    name: Some("Trait".to_owned()),
                    span: None,
                    visibility: Visibility::Public,
                    docs: None,
                    links: FxHashMap::default(),
                    attrs: Vec::new(),
                    deprecation: None,
                    stability: None,
                    const_stability: None,
                    inner: ItemEnum::Trait(Trait {
                        is_auto: false,
                        is_unsafe: false,
                        is_dyn_compatible: true,
                        items: vec![item_id],
                        generics: Generics { params: vec![], where_predicates: vec![] },
                        bounds: vec![],
                        implementations: vec![],
                    }),
                },
            ),
            (
                item_id,
                Item {
                    id: item_id,
                    crate_id: 0,
                    name: Some("TraitItem".to_owned()),
                    span: None,
                    visibility: Visibility::Public,
                    docs: None,
                    links: FxHashMap::default(),
                    attrs: Vec::new(),
                    deprecation: None,
                    stability: None,
                    const_stability: None,
                    inner,
                },
            ),
        ]),
        paths: FxHashMap::default(),
        external_crates: FxHashMap::default(),
        target: rustdoc_json_types::Target { triple: "".to_string(), target_features: vec![] },
        format_version: FORMAT_VERSION,
    }
}

#[test]
fn errors_on_default_unstable_without_function_body() {
    let krate = krate_with_trait_item(ItemEnum::Function(Function {
        sig: FunctionSignature { inputs: vec![], output: None, is_c_variadic: false },
        generics: Generics { params: vec![], where_predicates: vec![] },
        header: FunctionHeader {
            is_const: false,
            is_unsafe: false,
            is_async: false,
            abi: Abi::Rust,
        },
        has_body: false,
        default_unstable: Some(Box::new(ProvidedDefaultUnstable { feature: "feature".to_owned() })),
    }));

    check(
        &krate,
        &[Error {
            id: Id(2),
            kind: ErrorKind::Custom(
                "`default_unstable` must be `None` when `has_body == false`, but \
                 function item id 2 had `default_unstable` with feature `feature`"
                    .to_owned(),
            ),
        }],
    );
}

#[test]
fn errors_on_default_unstable_without_assoc_const_value() {
    let krate = krate_with_trait_item(ItemEnum::AssocConst {
        type_: Type::Primitive("usize".to_owned()),
        value: None,
        default_unstable: Some(Box::new(ProvidedDefaultUnstable { feature: "feature".to_owned() })),
    });

    check(
        &krate,
        &[Error {
            id: Id(2),
            kind: ErrorKind::Custom(
                "`default_unstable` must be `None` when `value` is `None`, but \
                 assoc const id 2 had `default_unstable` with feature `feature`"
                    .to_owned(),
            ),
        }],
    );
}

#[test]
fn errors_on_default_unstable_without_assoc_type_default() {
    let krate = krate_with_trait_item(ItemEnum::AssocType {
        generics: Generics { params: vec![], where_predicates: vec![] },
        bounds: vec![],
        type_: None,
        default_unstable: Some(Box::new(ProvidedDefaultUnstable { feature: "feature".to_owned() })),
    });

    check(
        &krate,
        &[Error {
            id: Id(2),
            kind: ErrorKind::Custom(
                "`default_unstable` must be `None` when `type_` is `None`, but \
                 assoc type id 2 had `default_unstable` with feature `feature`"
                    .to_owned(),
            ),
        }],
    );
}

#[test]
#[should_panic = "LOCAL_CRATE_ID is wrong"]
fn checks_local_crate_id_is_correct() {
    let krate = Crate {
        root: Id(0),
        crate_version: None,
        includes_private: false,
        index: FxHashMap::from_iter([(
            Id(0),
            Item {
                id: Id(0),
                crate_id: LOCAL_CRATE_ID.wrapping_add(1),
                name: Some("irrelavent".to_owned()),
                span: None,
                visibility: Visibility::Public,
                docs: None,
                links: FxHashMap::default(),
                attrs: Vec::new(),
                deprecation: None,
                stability: None,
                const_stability: None,
                inner: ItemEnum::Module(Module {
                    is_crate: true,
                    items: vec![],
                    is_stripped: false,
                }),
            },
        )]),
        paths: FxHashMap::default(),
        external_crates: FxHashMap::default(),
        target: rustdoc_json_types::Target { triple: "".to_string(), target_features: vec![] },
        format_version: FORMAT_VERSION,
    };
    check(&krate, &[]);
}
