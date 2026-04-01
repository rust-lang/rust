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

    // Check that all help options can be invoked at once
    let codegen_help = bare_rustc().arg("-Chelp").run().stdout_utf8();
    let unstable_help = bare_rustc().arg("-Zhelp").run().stdout_utf8();
    let lints_help = bare_rustc().arg("-Whelp").run().stdout_utf8();
    let expected_all = format!("{help}{codegen_help}{unstable_help}{lints_help}");
    let all_help = bare_rustc().args(["--help", "-Chelp", "-Zhelp", "-Whelp"]).run().stdout_utf8();
    diff()
        .expected_text(
            "(rustc --help && rustc -Chelp && rustc -Zhelp && rustc -Whelp)",
            &expected_all,
        )
        .actual_text("(rustc --help -Chelp -Zhelp -Whelp)", &all_help)
        .run();

    // Check that the ordering of help options is respected
    // Note that this is except for `-Whelp`, which always comes last
    let expected_ordered_help = format!("{unstable_help}{codegen_help}{help}{lints_help}");
    let ordered_help =
        bare_rustc().args(["-Whelp", "-Zhelp", "-Chelp", "--help"]).run().stdout_utf8();
    diff()
        .expected_text(
            "(rustc -Whelp && rustc -Zhelp && rustc -Chelp && rustc --help)",
            &expected_ordered_help,
        )
        .actual_text("(rustc -Whelp -Zhelp -Chelp --help)", &ordered_help)
        .run();

    // Test that `rustc --help` does not suppress invalid flag errors
    let help = bare_rustc().arg("--help --invalid-flag").run_fail().stdout_utf8();
}
