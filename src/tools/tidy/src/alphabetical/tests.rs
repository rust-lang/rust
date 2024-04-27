use super::*;
use std::io::Write;
use std::str::from_utf8;

fn test(lines: &str, name: &str, expected_msg: &str, expected_bad: bool) {
    let mut actual_msg = Vec::new();
    let mut actual_bad = false;
    let mut err = |args: &_| {
        write!(&mut actual_msg, "{args}")?;
        Ok(())
    };
    check_lines(&name, lines.lines().enumerate(), &mut err, &mut actual_bad);
    assert_eq!(expected_msg, from_utf8(&actual_msg).unwrap());
    assert_eq!(expected_bad, actual_bad);
}

fn good(lines: &str) {
    test(lines, "good", "", false);
}

fn bad(lines: &str, expected_msg: &str) {
    test(lines, "bad", expected_msg, true);
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
    bad(lines, "bad:4: line not in alphabetical order");
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
    bad(lines, "bad:4: line not in alphabetical order");
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
    bad(lines, "bad:4: line not in alphabetical order");
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
    bad(lines, "bad:4: line not in alphabetical order");
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
    bad(lines, "bad:7: line not in alphabetical order");
}

#[test]
fn test_double_start() {
    let lines = "\
        tidy-alphabetical-start
        abc
        tidy-alphabetical-start
    ";
    bad(lines, "bad:3 found `tidy-alphabetical-start` expecting `tidy-alphabetical-end`");
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
    bad(lines, "bad: reached end of file expecting `tidy-alphabetical-end`");
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
