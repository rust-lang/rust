// The flag `--output-format` is unauthorized on beta and stable releases, which led
// to confusion for maintainers doing testing on nightly. Tying it to an unstable flag
// elucidates this, and this test checks that `--output-format` cannot be passed on its
// own.
// See https://github.com/rust-lang/rust/pull/82497

use run_make_support::{diff, rustdoc};

fn main() {
    let out = rustdoc().output_format("json").input("x.html").run_fail().stderr_utf8();
    diff().expected_file("output-format-json.stderr").actual_text("actual-json", out).run();
}
