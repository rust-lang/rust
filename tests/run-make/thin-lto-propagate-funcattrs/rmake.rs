//! Check that the ThinLTO thin link propagates function attributes inferred in
//! other modules.
//!
//! `aux_callee` is `#[inline(never)]` and defined in another crate, so `caller`
//! only sees its declaration; the callee's summary is the only way the thin
//! link can learn that the call cannot unwind.

//@ needs-target-std
//@ needs-unwind
//@ ignore-cross-compile
// FIXME: GCC does not support ThinLTO.
//@ ignore-backends: gcc

use run_make_support::{llvm_dis, rfs, rustc};

fn main() {
    rustc()
        .input("aux.rs")
        .crate_name("aux")
        .crate_type("rlib")
        .opt_level("2")
        .arg("-Cembed-bitcode=yes")
        .run();
    rustc()
        .input("main.rs")
        .crate_name("main")
        .extern_("aux", "libaux.rlib")
        .opt_level("2")
        .arg("-Clto=thin")
        .arg("-Ccodegen-units=1")
        .arg("-Csave-temps")
        .output("main")
        .run();

    // The attributes are applied to the backend module, which `--emit=llvm-ir`
    // does not show. `-Csave-temps` keeps it as `...thin-lto-after-resolve.bc`.
    let mut resolved_modules = Vec::new();
    rfs::read_dir_entries(".", |path| {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else { return };
        if name.starts_with("main.main.") && name.ends_with(".thin-lto-after-resolve.bc") {
            resolved_modules.push(path.to_path_buf());
        }
    });
    assert!(
        !resolved_modules.is_empty(),
        "the thin link did not leave the module it resolved behind"
    );

    let mut seen_caller = false;
    for module in &resolved_modules {
        let ir = llvm_dis().arg("-o").arg("-").arg(module).run().stdout_utf8();
        if let Some(attrs) = attributes_of(&ir, "@caller(") {
            assert!(
                attrs.contains("nounwind"),
                "`caller` is missing `nounwind`; attributes:\n{attrs}"
            );
            seen_caller = true;
        }
    }
    assert!(seen_caller, "could not find the definition of `caller` in the resolved module");
}

/// The attribute group of the first definition of `name`.
fn attributes_of<'a>(ir: &'a str, name: &str) -> Option<&'a str> {
    let def = ir.lines().find(|line| line.starts_with("define") && line.contains(name))?;
    let group = def.split_whitespace().find_map(|token| token.strip_prefix('#'))?;
    ir.lines().find(|line| line.starts_with(&format!("attributes #{group} ")))
}
