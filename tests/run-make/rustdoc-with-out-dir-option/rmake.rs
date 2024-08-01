use run_make_support::{htmldocck, rustdoc};

fn main() {
    let out_dir = "rustdoc";
    rustdoc().input("src/lib.rs").crate_name("foobar").crate_type("lib").output(&out_dir).run();
    htmldocck().arg(out_dir).arg("src/lib.rs").run();
}
