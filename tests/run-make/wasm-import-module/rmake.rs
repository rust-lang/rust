//@ only-wasm32-wasip1

use std::collections::HashMap;

use run_make_support::{rfs, rustc, wasmparser};
use wasmparser::TypeRef::Func;

fn main() {
    rustc().input("foo.rs").target("wasm32-wasip1").run();
    rustc().input("bar.rs").target("wasm32-wasip1").arg("-Clto").opt().run();

    let file = rfs::read("bar.wasm");

    let mut imports = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::ImportSection(s) = payload {
            for i in s {
                let i = i.unwrap();
                imports.entry(i.module).or_insert(Vec::new()).push((i.name, i.ty));
            }
        }
    }

    let import = imports.remove("./dep");
    assert!(matches!(import.as_deref(), Some([("dep", Func(_))])), "bad import {:?}", import);
    let import = imports.remove("./me");
    assert!(matches!(import.as_deref(), Some([("me_in_dep", Func(_))])), "bad import {:?}", import);
}
