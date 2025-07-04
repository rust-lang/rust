//! Disassembly calling function for `wasm32` targets.

use crate::Function;
use std::collections::HashSet;

pub(crate) fn disassemble_myself() -> HashSet<Function> {
    // Use `std::env::args` to find the path to our executable. Assume the
    // environment is configured such that we can read that file. Read it and
    // use the `wasmprinter` crate to transform the binary to text, then search
    // the text for appropriately named functions.
    let me = std::env::args()
        .next()
        .expect("failed to find current wasm file");
    let output = wasmprinter::print_file(&me).unwrap();

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
