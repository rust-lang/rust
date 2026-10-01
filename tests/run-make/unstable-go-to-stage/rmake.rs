use run_make_support::{invalid_utf8_contains, invalid_utf8_not_contains, rustc};

fn main() {
    rustc().input("aux.rs").crate_type("rlib").emit("link,metadata").arg("-Zgo-to-stage=end").run();
    rustc()
        .input("lib.rs")
        .crate_type("rlib")
        .emit("dep-info")
        .arg("-Zbinary-dep-depinfo")
        .arg("-Zgo-to-stage=end")
        .run();
    invalid_utf8_contains("lib.d", "libaux.rmeta");
    invalid_utf8_not_contains("lib.d", "libaux.rlib");
}
