//! Disassembly calling function for `wasm32` targets.

use crate::Function;
use std::collections::HashSet;
use std::path::Path;

pub(crate) fn disassemble_myself() -> HashSet<Function> {
    // Use `std::env::args` to find the path to our executable. Assume the
    // environment is configured such that we can read that file. Read it and
    // use the `wasmprinter` crate to transform the binary to text, then search
    // the text for appropriately named functions.
    let me = std::env::args()
        .next()
        .expect("failed to find current wasm file");
    let me = Path::new(&me);
    let me = if me.exists() {
        // Old build-dir layout
        me.to_path_buf()
    } else {
        // Cargo's build-dir layout stores an artifact named
        // `<crate>-<hash>.wasm` at `<crate>/<hash>/out/<crate>-<hash>.wasm`.
        // The build directory is mounted as the WASI working directory, so
        // reconstruct that guest-visible path from the executable name.
        let file_name = me
            .file_name()
            .and_then(|name| name.to_str())
            .expect("current wasm file has a file name");
        let stem = file_name
            .strip_suffix(".wasm")
            .expect("current wasm file does not have a .wasm extension");
        let (crate_name, hash) = stem
            .rsplit_once('-')
            .expect("current wasm file name does not contain an artifact hash");
        Path::new(crate_name).join(hash).join("out").join(file_name)
    };
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
