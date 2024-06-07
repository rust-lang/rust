use run_make_support::{rustc, tmp_dir};

fn main() {
    let out = rustc()
        .link_arg("-lfoo")
        .link_arg("-lbar")
        .print("link-args")
        .input("empty.rs")
        .command_output();
    let out = String::from_utf8(out.stdout).unwrap();
    assert!(out.contains("lfoo"), "output doesn't contain `lfoo`");
    assert!(out.contains("lbar"), "output doesn't contain `lbar`");
}
