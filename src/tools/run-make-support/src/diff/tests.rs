#[cfg(test)]
mod tests {
    use crate::*;

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
            diff()
                .expected_text("EXPECTED_TEXT", expected)
                .actual_text("ACTUAL_TEXT", actual)
                .run();
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
}
