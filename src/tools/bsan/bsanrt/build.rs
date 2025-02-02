use std::env;
use std::path::Path;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("BSAN_HEADER_DIR").unwrap_or(env::var("OUT_DIR").unwrap());
    cbindgen::Builder::new()
        .with_config(cbindgen::Config::from_root_or_default(crate_dir.clone()))
        .with_crate(crate_dir.clone())
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(Path::new(&out_dir).join("bsanrt.h"));
}
