use crate::diff;

#[test]
fn test_diff() {
    let expected = "foo\nbar\nbaz\n";
    let actual = "foo\nbar\nbaz\n";
    diff().expected_text("EXPECTED_TEXT", expected).actual_text("ACTUAL_TEXT", actual).run();
}

#[test]
fn test_should_panic() {
    let expected = "foo\nbar\nbaz\n";
    let actual = "foo\nbaz\nbar\n";

    let output = std::panic::catch_unwind(|| {
        diff().expected_text("EXPECTED_TEXT", expected).actual_text("ACTUAL_TEXT", actual).run();
    })
    .unwrap_err();

    let expected_output = "\
test failed: `EXPECTED_TEXT` is different from `ACTUAL_TEXT`

--- EXPECTED_TEXT
+++ ACTUAL_TEXT
@@ -1,3 +1,3 @@
 foo
+baz
 bar
-baz
";

    assert_eq!(output.downcast_ref::<String>().unwrap(), expected_output);
}

#[test]
fn test_normalize() {
    let expected = "
running 2 tests
..

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in $TIME
";
    let actual = "
running 2 tests
..

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
";

    diff()
        .expected_text("EXPECTED_TEXT", expected)
        .actual_text("ACTUAL_TEXT", actual)
        .normalize(r#"finished in \d+\.\d+s"#, "finished in $$TIME")
        .run();
}
