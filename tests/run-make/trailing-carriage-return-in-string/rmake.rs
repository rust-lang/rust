//@ ignore-cross-compile

use run_make_support::{cwd, diff, rfs, rustc};

fn main() {
    let test_content = "fn main() {\n\
    // \\r\\n\n\
    let ok = \"This is \\\r\n a test\";\n\
    // \\r only\n\
    let bad = \"This is \\\r a test\";\n\
}\n";

    let test_path = cwd().join("trailing-carriage-return-in-string.rs");
    rfs::write(&test_path, test_content);

    let output = rustc().input(&test_path).ui_testing().run_fail().stderr_utf8();

    diff()
        .expected_file("trailing-carriage-return-in-string.stderr")
        .actual_text("stderr", &output)
        .normalize(r#"--> [^\n]+"#, "--> trailing-carriage-return-in-string.rs:LL:CC")
        .run();
}
