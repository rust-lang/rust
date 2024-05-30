use run_make_support::{rustc, rustdoc, Diff};

fn compare_outputs(args: &[&str]) {
    let rustc_output = String::from_utf8(rustc().args(args).run().stdout).unwrap();
    let rustdoc_output = String::from_utf8(rustdoc().args(args).run().stdout).unwrap();

    Diff::new().expected_text("rustc", rustc_output).actual_text("rustdoc", rustdoc_output).run();
}

fn main() {
    compare_outputs(&["-C", "help"]);
    compare_outputs(&["-Z", "help"]);
    compare_outputs(&["-C", "passes=list"]);
}
