//! Disassembly calling function for `wasm32` targets.
use wasm_bindgen::prelude::*;

use crate::Function;
use std::collections::HashSet;

#[wasm_bindgen(module = "child_process")]
extern "C" {
    #[wasm_bindgen(js_name = execFileSync)]
    fn exec_file_sync(cmd: &str, args: &js_sys::Array, opts: &js_sys::Object) -> Buffer;
}

#[wasm_bindgen(module = "buffer")]
extern "C" {
    type Buffer;
    #[wasm_bindgen(method, js_name = toString)]
    fn to_string(this: &Buffer) -> String;
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = require)]
    fn resolve(module: &str) -> String;
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    pub fn js_console_log(s: &str);
}

pub(crate) fn disassemble_myself() -> HashSet<Function> {
    use std::path::Path;
    ::console_error_panic_hook::set_once();
    // Our wasm module in the wasm-bindgen test harness is called
    // "wasm-bindgen-test_bg". When running in node this is actually a shim JS
    // file. Ask node where that JS file is, and then we use that with a wasm
    // extension to find the wasm file itself.
    let js_shim = resolve("wasm-bindgen-test_bg");
    let js_shim = Path::new(&js_shim).with_extension("wasm");

    // Execute `wasm2wat` synchronously, waiting for and capturing all of its
    // output. Note that we pass in a custom `maxBuffer` parameter because we're
    // generating a ton of output that needs to be buffered.
    let args = js_sys::Array::new();
    args.push(&js_shim.display().to_string().into());
    args.push(&"--enable-simd".into());
    let opts = js_sys::Object::new();
    js_sys::Reflect::set(&opts, &"maxBuffer".into(), &(200 * 1024 * 1024).into())
        .unwrap();
    let output = exec_file_sync("wasm2wat", &args, &opts).to_string();

    let mut ret: HashSet<Function> = HashSet::new();
    let mut lines = output.lines().map(|s| s.trim());
    while let Some(line) = lines.next() {
        // If this isn't a function, we don't care about it.
        if !line.starts_with("(func ") {
            continue;
        }

        let mut function = Function {
            name: String::new(),
            instrs: Vec::new(),
        };

        // Empty functions will end in `))` so there's nothing to do, otherwise
        // we'll have a bunch of following lines which are instructions.
        //
        // Lines that have an imbalanced `)` mark the end of a function.
        if !line.ends_with("))") {
            while let Some(line) = lines.next() {
                function.instrs.push(line.to_string());
                if !line.starts_with("(") && line.ends_with(")") {
                    break;
                }
            }
        }
        // The second element here split on whitespace should be the name of
        // the function, skipping the type/params/results
        function.name = line.split_whitespace().nth(1).unwrap().to_string();
        if function.name.starts_with("$") {
            function.name = function.name[1..].to_string()
        }

        if !function.name.contains("stdarch_test_shim") {
            continue;
        }

        assert!(ret.insert(function));
    }
    return ret;
}
