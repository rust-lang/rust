use std::path::Path;

use crate::alphabetical::check_lines;
use crate::diagnostics::{TidyCtx, TidyFlags};

#[track_caller]
fn test(lines: &str, name: &str, expected_msg: &str, expected_bad: bool) {
    let tidy_ctx = TidyCtx::new(Path::new("/"), false, TidyFlags::default());
    let mut check = tidy_ctx.start_check("alphabetical-test");
    check_lines(Path::new(name), lines, &tidy_ctx, &mut check);

    assert_eq!(expected_bad, check.is_bad());
    let errors = check.get_errors();
    if expected_bad {
        assert_eq!(errors.len(), 1);
        assert_eq!(expected_msg, errors[0]);
    } else {
        assert!(errors.is_empty());
    }
}

#[track_caller]
fn good(lines: &str) {
    test(lines, "good", "", false);
}

#[track_caller]
fn bad(lines: &str, expected_msg: &str) {
    test(lines, "bad", expected_msg, true);
}

#[track_caller]
fn bless_test(before: &str, after: &str) {
    // NB: convert to a temporary *path* (closing the file), so that `check_lines` can then
    //     atomically replace the file with a blessed version (on windows that requires the file
    //     to not be open)
    let temp_path = tempfile::Builder::new().tempfile().unwrap().into_temp_path();
    std::fs::write(&temp_path, before).unwrap();

    let tidy_ctx = TidyCtx::new(Path::new("/"), false, TidyFlags::new(&["--bless".to_owned()]));

    let mut check = tidy_ctx.start_check("alphabetical-test");
    check_lines(&temp_path, before, &tidy_ctx, &mut check);

    assert!(!check.is_bad());
    let new = std::fs::read_to_string(temp_path).unwrap();
    assert_eq!(new, after);

    good(&new);
}

#[test]
fn test_no_markers() {
    let lines = "\
        def
        abc
        xyz
    ";
    good(lines);
}

#[test]
fn test_rust_good() {
    let lines = "\
        // tidy-alphabetical-start
        abc
        def
        xyz
        // tidy-alphabetical-end"; // important: end marker on last line
    good(lines);
}

#[test]
fn test_complex_good() {
    let lines = "\
        zzz

        // tidy-alphabetical-start
        abc
        // Rust comments are ok
        def
        # TOML comments are ok
        xyz
        // tidy-alphabetical-end

        # tidy-alphabetical-start
        foo(abc);
        // blank lines are ok

        // split line gets joined
        foo(
            def
        );

        foo(xyz);
        # tidy-alphabetical-end

        % tidy-alphabetical-start
        abc
            ignored_due_to_different_indent
        def
        % tidy-alphabetical-end

        aaa
    ";
    good(lines);
}

#[test]
fn test_rust_bad() {
    let lines = "\
        // tidy-alphabetical-start
        abc
        xyz
        def
        // tidy-alphabetical-end
    ";
    bad(lines, "bad:2: line not in alphabetical order (tip: use --bless to sort this list)");
}

#[test]
fn test_toml_bad() {
    let lines = "\
        # tidy-alphabetical-start
        abc
        xyz
        def
        # tidy-alphabetical-end
    ";
    bad(lines, "bad:2: line not in alphabetical order (tip: use --bless to sort this list)");
}

#[test]
fn test_features_bad() {
    // Even though lines starting with `#` are treated as comments, lines
    // starting with `#!` are an exception.
    let lines = "\
        tidy-alphabetical-start
        #![feature(abc)]
        #![feature(xyz)]
        #![feature(def)]
        tidy-alphabetical-end
    ";
    bad(lines, "bad:2: line not in alphabetical order (tip: use --bless to sort this list)");
}

#[test]
fn test_indent_bad() {
    // All lines are indented the same amount, and so are checked.
    let lines = "\
        $ tidy-alphabetical-start
            abc
            xyz
            def
        $ tidy-alphabetical-end
    ";
    bad(lines, "bad:2: line not in alphabetical order (tip: use --bless to sort this list)");
}

#[test]
fn test_split_bad() {
    let lines = "\
        || tidy-alphabetical-start
        foo(abc)
        foo(
            xyz
        )
        foo(
            def
        )
        && tidy-alphabetical-end
    ";
    bad(lines, "bad:3: line not in alphabetical order (tip: use --bless to sort this list)");
}

#[test]
fn test_double_start() {
    let lines = "\
        tidy-alphabetical-start
        abc
        tidy-alphabetical-start
    ";
    bad(lines, "bad:0 `tidy-alphabetical-start` without a matching `tidy-alphabetical-end`");
}

#[test]
fn test_missing_start() {
    let lines = "\
        abc
        tidy-alphabetical-end
        abc
    ";
    bad(lines, "bad:2 found `tidy-alphabetical-end` expecting `tidy-alphabetical-start`");
}

#[test]
fn test_missing_end() {
    let lines = "\
        tidy-alphabetical-start
        abc
    ";
    bad(lines, "bad:0 `tidy-alphabetical-start` without a matching `tidy-alphabetical-end`");
}

#[test]
fn test_double_end() {
    let lines = "\
        tidy-alphabetical-start
        abc
        tidy-alphabetical-end
        def
        tidy-alphabetical-end
    ";
    bad(lines, "bad:5 found `tidy-alphabetical-end` expecting `tidy-alphabetical-start`");
}

#[test]
fn test_numeric_good() {
    good(
        "\
        # tidy-alphabetical-start
        rustc_ast = { path = \"../rustc_ast\" }
        rustc_ast_lowering = { path = \"../rustc_ast_lowering\" }
        # tidy-alphabetical-end
    ",
    );

    good(
        "\
        # tidy-alphabetical-start
        fp-armv8
        fp16
        # tidy-alphabetical-end
    ",
    );

    good(
        "\
        # tidy-alphabetical-start
        item1
        item2
        item10
        # tidy-alphabetical-end
    ",
    );

    good(
        "\
        # tidy-alphabetical-start
        foo
        foo_
        # tidy-alphabetical-end
    ",
    );

    good(
        "\
        # tidy-alphabetical-start
        foo-bar
        foo_bar
        # tidy-alphabetical-end
    ",
    );

    good(
        "\
        # tidy-alphabetical-start
        sme-lutv2
        sme2
        # tidy-alphabetical-end
    ",
    );

    good(
        "\
        # tidy-alphabetical-start
        v5te
        v6
        v6k
        v6t2
        # tidy-alphabetical-end
    ",
    );

    good(
        "\
        # tidy-alphabetical-start
        zve64d
        zve64f
        # tidy-alphabetical-end
    ",
    );

    // Case is significant.
    good(
        "\
        # tidy-alphabetical-start
        _ZYXW
        _abcd
        # tidy-alphabetical-end
    ",
    );

    good(
        "\
        # tidy-alphabetical-start
        v0
        v00
        v000
        # tidy-alphabetical-end
    ",
    );

    good(
        "\
        # tidy-alphabetical-start
        w005s09t
        w5s009t
        # tidy-alphabetical-end
    ",
    );

    good(
        "\
        # tidy-alphabetical-start
        v0s
        v00t
        # tidy-alphabetical-end
    ",
    );
}

#[test]
fn test_numeric_bad() {
    let lines = "\
        # tidy-alphabetical-start
        item1
        item10
        item2
        # tidy-alphabetical-end
    ";
    bad(lines, "bad:2: line not in alphabetical order (tip: use --bless to sort this list)");

    let lines = "\
        # tidy-alphabetical-start
        zve64f
        zve64d
        # tidy-alphabetical-end
    ";
    bad(lines, "bad:1: line not in alphabetical order (tip: use --bless to sort this list)");

    let lines = "\
        # tidy-alphabetical-start
        000
        00
        # tidy-alphabetical-end
    ";
    bad(lines, "bad:1: line not in alphabetical order (tip: use --bless to sort this list)");
}

#[test]
fn multiline() {
    let lines = "\
        tidy-alphabetical-start
        (b,
         a);
        (
          b,
          a
        );
        tidy-alphabetical-end
    ";
    good(lines);

    let lines = "\
        tidy-alphabetical-start
        (
          b,
          a
        );
        (b,
         a);
        tidy-alphabetical-end
    ";
    good(lines);

    let lines = "\
        tidy-alphabetical-start
        (c,
         a);
        (
          b,
          a
        );
        tidy-alphabetical-end
    ";
    bad(lines, "bad:1: line not in alphabetical order (tip: use --bless to sort this list)");

    let lines = "\
        tidy-alphabetical-start
        (
          c,
          a
        );
        (b,
         a);
        tidy-alphabetical-end
    ";
    bad(lines, "bad:1: line not in alphabetical order (tip: use --bless to sort this list)");

    let lines = "\
        force_unwind_tables: Option<bool> = (None, parse_opt_bool, [TRACKED],
             'force use of unwind tables'),
        incremental: Option<String> = (None, parse_opt_string, [UNTRACKED],
            'enable incremental compilation'),
    ";
    good(lines);
}

#[test]
fn bless_smoke() {
    let before = "\
        tidy-alphabetical-start
        08
        1
        11
        03
        tidy-alphabetical-end
    ";
    let after = "\
        tidy-alphabetical-start
        1
        03
        08
        11
        tidy-alphabetical-end
    ";

    bless_test(before, after);
}

#[test]
fn bless_multiline() {
    let before = "\
        tidy-alphabetical-start
        08 {
             z}
        08 {
           x
        }
        1
        08 {y}
        02
        11 (
          0
        )
        03
            addition
    notaddition
        tidy-alphabetical-end
    ";
    let after = "\
        tidy-alphabetical-start
        1
        02
        03
            addition
        08 {
           x
        }
        08 {y}
        08 {
             z}
        11 (
          0
        )
    notaddition
        tidy-alphabetical-end
    ";

    bless_test(before, after);
}

#[test]
fn bless_funny_numbers() {
    // Because `2` is indented it gets merged into one entry with `1` and gets
    // interpreted by version sort as `12`, which is greater than `3`.
    //
    // This is neither a wanted nor an unwanted behavior, this test just checks
    // that it hasn't changed.

    let before = "\
        tidy-alphabetical-start
        1
          2
        3
        tidy-alphabetical-end
    ";
    let after = "\
        tidy-alphabetical-start
        3
        1
          2
        tidy-alphabetical-end
    ";

    bless_test(before, after);
}
