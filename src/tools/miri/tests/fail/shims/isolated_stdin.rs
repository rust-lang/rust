//@ignore-target: windows # FIXME: stdin does not work on Windows
//@error-in-other-file: `read` from stdin not available when isolation is enabled
//@normalize-stderr-test: "src/sys/.*\.rs" -> "$$FILE"
//@normalize-stderr-test: "\nLL \| .*" -> ""
//@normalize-stderr-test: "\| +[|_^]+" -> "| ^"
//@normalize-stderr-test: "\n *= note:.*" -> ""
use std::io::{self, Read};

fn main() {
    let mut bytes = [0u8; 512];
    io::stdin().read(&mut bytes).unwrap();
}
