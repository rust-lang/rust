use std::path;

use run_make_support::rustdoc_json_types::{Crate, ItemEnum, Path, Type, TypeAlias};
use run_make_support::{cwd, rfs, rust_lib_name, rustc, rustdoc, serde_json};

#[track_caller]
fn canonicalize(p: &path::Path) -> path::PathBuf {
    std::fs::canonicalize(p).expect("path should be canonicalizeable")
}

fn main() {
    rustc().input("trans_dep.rs").edition("2024").crate_type("lib").run();

    rustc()
        .input("dep.rs")
        .edition("2024")
        .crate_type("lib")
        .extern_("trans_dep", rust_lib_name("trans_dep"))
        .run();

    rustdoc()
        .input("entry.rs")
        .edition("2024")
        .output_format("json")
        .library_search_path(cwd())
        .extern_("dep", rust_lib_name("dep"))
        .arg("-Zunstable-options")
        .run();

    let bytes = rfs::read("doc/entry.json");

    let krate: Crate = serde_json::from_slice(&bytes).expect("output should be valid json");

    let root_item = &krate.index[&krate.root];
    let ItemEnum::Module(root_mod) = &root_item.inner else { panic!("expected ItemEnum::Module") };

    assert_eq!(root_mod.items.len(), 2);

    let items = root_mod.items.iter().map(|id| &krate.index[id]).collect::<Vec<_>>();

    let from_dep = items
        .iter()
        .filter(|item| item.name.as_deref() == Some("FromDep"))
        .next()
        .expect("there should be en item called FromDep");

    let from_trans_dep = items
        .iter()
        .filter(|item| item.name.as_deref() == Some("FromTransDep"))
        .next()
        .expect("there should be en item called FromDep");

    let ItemEnum::TypeAlias(TypeAlias {
        type_: Type::ResolvedPath(Path { id: from_dep_id, .. }),
        ..
    }) = &from_dep.inner
    else {
        panic!("Expected FromDep to be a TypeAlias");
    };

    let ItemEnum::TypeAlias(TypeAlias {
        type_: Type::ResolvedPath(Path { id: from_trans_dep_id, .. }),
        ..
    }) = &from_trans_dep.inner
    else {
        panic!("Expected FromDep to be a TypeAlias");
    };

    assert_eq!(krate.index.get(from_dep_id), None);
    assert_eq!(krate.index.get(from_trans_dep_id), None);

    let from_dep_externalinfo = &krate.paths[from_dep_id];
    let from_trans_dep_externalinfo = &krate.paths[from_trans_dep_id];

    let dep_crate_id = from_dep_externalinfo.crate_id;
    let trans_dep_crate_id = from_trans_dep_externalinfo.crate_id;

    let dep = &krate.external_crates[&dep_crate_id];
    let trans_dep = &krate.external_crates[&trans_dep_crate_id];

    assert_eq!(dep.name, "dep");
    assert_eq!(trans_dep.name, "trans_dep");

    assert_eq!(canonicalize(&dep.path), canonicalize(&cwd().join(rust_lib_name("dep"))));
    assert_eq!(
        canonicalize(&trans_dep.path),
        canonicalize(&cwd().join(rust_lib_name("trans_dep")))
    );
}
