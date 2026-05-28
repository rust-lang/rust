use run_make_support::{Diff, rustc, rustdoc};

fn compare_outputs(args: &[&str]) {
    let rustc_output = rustc().args(args).run().stdout_utf8();
    let rustdoc_output = rustdoc().args(args).run().stdout_utf8();

    Diff::new().expected_text("rustc", rustc_output).actual_text("rustdoc", rustdoc_output).run();
}

fn main() {
    compare_outputs(&["-C", "help"]);
    compare_outputs(&["-Z", "help"]);
    compare_outputs(&["-C", "passes=list"]);
}
