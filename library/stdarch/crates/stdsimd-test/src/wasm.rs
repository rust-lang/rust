//! Disassembly calling function for `wasm32` targets.
use wasm_bindgen::prelude::*;

use ::*;

#[wasm_bindgen(module = "child_process")]
extern "C" {
    #[wasm_bindgen(js_name = execSync)]
    fn exec_sync(cmd: &str) -> Buffer;
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

pub(crate) fn disassemble_myself() -> HashMap<String, Vec<Function>> {
    use std::path::Path;
    ::console_error_panic_hook::set_once();
    // Our wasm module in the wasm-bindgen test harness is called
    // "wasm-bindgen-test_bg". When running in node this is actually a shim JS
    // file. Ask node where that JS file is, and then we use that with a wasm
    // extension to find the wasm file itself.
    let js_shim = resolve("wasm-bindgen-test_bg");
    let js_shim = Path::new(&js_shim).with_extension("wasm");

    // Execute `wasm2wat` synchronously, waiting for and capturing all of its
    // output.
    let output =
        exec_sync(&format!("wasm2wat {}", js_shim.display())).to_string();

    let mut ret: HashMap<String, Vec<Function>> = HashMap::new();
    let mut lines = output.lines().map(|s| s.trim());
    while let Some(line) = lines.next() {
        // If we found the table of function pointers, fill in the known
        // address for all our `Function` instances
        if line.starts_with("(elem") {
            let mut parts = line.split_whitespace().skip(3);
            let offset = parts.next()
                .unwrap()
                .trim_end_matches(")")
                .parse::<usize>()
                .unwrap();
            for (i, name) in parts.enumerate() {
                let name = name.trim_end_matches(")");
                for f in ret.get_mut(name).expect("ret.get_mut(name) failed") {
                    f.addr = Some(i + offset);
                }
            }
            continue;
        }

        // If this isn't a function, we don't care about it.
        if !line.starts_with("(func ") {
            continue;
        }

        let mut function = Function {
            instrs: Vec::new(),
            addr: None,
        };

        // Empty functions will end in `))` so there's nothing to do, otherwise
        // we'll have a bunch of following lines which are instructions.
        //
        // Lines that have an imbalanced `)` mark the end of a function.
        if !line.ends_with("))") {
            while let Some(line) = lines.next() {
                function.instrs.push(Instruction {
                    parts: line
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect(),
                });
                if !line.starts_with("(") && line.ends_with(")") {
                    break;
                }
            }
        }

        // The second element here split on whitespace should be the name of
        // the function, skipping the type/params/results
        ret.entry(line.split_whitespace().nth(1).unwrap().to_string())
            .or_insert(Vec::new())
            .push(function);
    }
    return ret;
}
