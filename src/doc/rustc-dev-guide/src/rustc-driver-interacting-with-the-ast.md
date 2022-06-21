# Example: Type checking through `rustc_interface`

`rustc_interface` allows you to interact with Rust code at various stages of compilation.

## Getting the type of an expression

To get the type of an expression, use the `global_ctxt` to get a `TyCtxt`.
The following was tested with <!-- date: 2022-06 --> `nightly-2022-06-05`
(see [here][example] for the complete example):

[example]: https://github.com/rust-lang/rustc-dev-guide/blob/master/examples/rustc-driver-interacting-with-the-ast.rs

```rust
let config = rustc_interface::Config {
    input: config::Input::Str {
        name: source_map::FileName::Custom("main.rs".to_string()),
        input: "fn main() { let message = \"Hello, world!\"; println!(\"{}\", message); }"
            .to_string(),
    },
    /* other config */
};
rustc_interface::run_compiler(config, |compiler| {
    compiler.enter(|queries| {
        // Analyze the crate and inspect the types under the cursor.
        queries.global_ctxt().unwrap().take().enter(|tcx| {
            // Every compilation contains a single crate.
            let hir_krate = tcx.hir();
            // Iterate over the top-level items in the crate, looking for the main function.
            for id in hir_krate.items() {
                let item = hir_krate.item(id);
                // Use pattern-matching to find a specific node inside the main function.
                if let rustc_hir::ItemKind::Fn(_, _, body_id) = item.kind {
                    let expr = &tcx.hir().body(body_id).value;
                    if let rustc_hir::ExprKind::Block(block, _) = expr.kind {
                        if let rustc_hir::StmtKind::Local(local) = block.stmts[0].kind {
                            if let Some(expr) = local.init {
                                let hir_id = expr.hir_id; // hir_id identifies the string "Hello, world!"
                                let def_id = tcx.hir().local_def_id(item.hir_id()); // def_id identifies the main function
                                let ty = tcx.typeck(def_id).node_type(hir_id);
                                println!("{:?}: {:?}", expr, ty);
                            }
                        }
                    }
                }
            }
        })
    });
});
```
