// Tests `rustc --help` and similar invocations against snapshots and each other.

use run_make_support::{bare_rustc, diff, similar};

fn main() {
    // `rustc --help`
    let help = bare_rustc().arg("--help").run().stdout_utf8();
    diff().expected_file("help.stdout").actual_text("(rustc --help)", &help).run();

    // `rustc` should be the same as `rustc --help`
    let bare = bare_rustc().run().stdout_utf8();
    diff().expected_text("(rustc --help)", &help).actual_text("(rustc)", &bare).run();

    // `rustc --help -v` should give a similar but longer help message
    let help_v = bare_rustc().arg("--help").arg("-v").run().stdout_utf8();
    diff().expected_file("help-v.stdout").actual_text("(rustc --help -v)", &help_v).run();

    // Check the diff between `rustc --help` and `rustc --help -v`.
    let help_v_diff = similar::TextDiff::from_lines(&help, &help_v).unified_diff().to_string();
    diff().expected_file("help-v.diff").actual_text("actual", &help_v_diff).run();
}
