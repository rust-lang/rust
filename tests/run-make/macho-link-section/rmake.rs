//! Test that various Mach-O `#[link_section]` values are parsed and passed on correctly by codegen
//! backends.
//@ only-apple
use run_make_support::{llvm_objdump, rustc};

fn main() {
    rustc().input("foo.rs").crate_type("lib").arg("--emit=obj").run();

    let stdout =
        llvm_objdump().arg("--macho").arg("--private-headers").input("foo.o").run().stdout_utf8();

    let expected = [
        ("__TEXT", "custom_code", "S_REGULAR", "PURE_INSTRUCTIONS"),
        ("__DATA", "__mod_init_func", "S_MOD_INIT_FUNC_POINTERS", "(none)"),
        (
            "__DATA",
            "all_attributes",
            "S_REGULAR",
            "PURE_INSTRUCTIONS NO_TOC STRIP_STATIC_SYMS \
             NO_DEAD_STRIP LIVE_SUPPORT SELF_MODIFYING_CODE DEBUG",
        ),
    ];

    for (segment, section, section_type, section_attributes) in expected {
        let mut found = false;
        // Skip header.
        for section_info in stdout.split("Section").skip(1) {
            if section_info.contains(&format!("segname {segment}"))
                && section_info.contains(&format!("sectname {section}"))
            {
                assert!(
                    section_info.contains(&format!("type {section_type}")),
                    "should have type {section_type:?}"
                );
                assert!(
                    section_info.contains(&format!("attributes {section_attributes}\n")),
                    "should have attributes {section_attributes:?}"
                );
                found = true;
            }
        }

        if !found {
            panic!("could not find section {section} in binary");
        }
    }
}
