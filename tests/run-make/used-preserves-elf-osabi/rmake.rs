//! Verifies that plain `#[used]` preserves `ELFOSABI_NONE` on non-GNU ELF targets.

//@ needs-llvm-components: x86 aarch64

use std::fs;
use std::path::Path;

use run_make_support::{rustc, rustc_minicore};

fn main() {
    let targets = [("x86_64-unknown-none", "x86_64"), ("aarch64-unknown-none", "aarch64")];

    for (target, name) in targets {
        let minicore = format!("libminicore-{name}.rlib");
        let object = format!("used-{name}.o");

        rustc_minicore().target(target).output(&minicore).run();

        rustc()
            .input("used.rs")
            .crate_type("lib")
            .target(target)
            .extern_("minicore", Path::new(&minicore))
            .emit("obj")
            .output(&object)
            .run();

        let bytes = fs::read(&object).unwrap();

        assert_eq!(&bytes[..4], b"\x7fELF", "{target} did not produce ELF");
        assert_eq!(bytes[7], 0, "{target}: expected ELFOSABI_NONE, found OSABI {}", bytes[7]);
    }
}
