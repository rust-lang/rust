use super::*;

// If vertical_trim trim first and last line.
#[test]
fn trim_vertically_first_or_line() {
    // Accepted cases

    let inp = &["*********************************", "* This is a module to do foo job."];
    let out = &["* This is a module to do foo job."];
    assert_eq!(vertical_trim(inp), out);

    let inp = &["* This is a module to do foo job.", "*********************************"];
    let out = &["* This is a module to do foo job."];
    assert_eq!(vertical_trim(inp), out);

    let inp = &[
        "*********************************",
        "* This is a module to do foo job.",
        "*********************************",
    ];
    let out = &["* This is a module to do foo job."];
    assert_eq!(vertical_trim(inp), out);

    let inp = &[
        "***********************",
        "* This is a module to do foo job.",
        "*********************************",
    ];
    let out = &["* This is a module to do foo job."];
    assert_eq!(vertical_trim(inp), out);

    let inp = &[
        "**************************",
        " * one two three four five six seven",
        " ****************",
    ];
    let out = &[" * one two three four five six seven"];
    assert_eq!(vertical_trim(inp), out);

    let inp = &["", " * one two three four five", " "];
    let out = &[" * one two three four five"];
    assert_eq!(vertical_trim(inp), out);

    // Non-accepted cases

    let inp = &["\t  *********************** \t", "* This is a module to do foo job."];
    let out = &["\t  *********************** \t", "* This is a module to do foo job."];
    assert_eq!(vertical_trim(inp), out);

    // More than one space indentation.
    let inp = &[
        "******************************",
        "  * This is a module to do foo job.",
        "  **************",
    ];
    let out = &["  * This is a module to do foo job.", "  **************"];
    assert_eq!(vertical_trim(inp), out);
}

// Trim consecutive empty lines. Break if meet a non-empty line.
#[test]
fn trim_vertically_empty_lines_forward() {
    let inp = &["    ", "    \t    \t  ", " * One two three four five six seven eight nine ten."];
    let out = &[" * One two three four five six seven eight nine ten."];
    assert_eq!(vertical_trim(inp), out);

    let inp = &[
        "    ",
        " * One two three four five six seven eight nine ten.",
        "    \t    \t  ",
        " * One two three four five six seven eight nine ten.",
    ];
    let out = &[
        " * One two three four five six seven eight nine ten.",
        "    \t    \t  ",
        " * One two three four five six seven eight nine ten.",
    ];
    assert_eq!(vertical_trim(inp), out);
}

// Trim consecutive empty lines bottom-top. Break if meet a non-empty line.
#[test]
fn trim_vertically_empty_lines_backward() {
    let inp = &[" * One two three four five six seven eight nine ten.", "    ", "    \t    \t  "];
    let out = &[" * One two three four five six seven eight nine ten."];
    assert_eq!(vertical_trim(inp), out);

    let inp = &[
        " * One two three four five six seven eight nine ten.",
        "    ",
        " * One two three four five six seven eight nine ten.",
        "    \t    \t  ",
    ];
    let out = &[
        " * One two three four five six seven eight nine ten.",
        "    ",
        " * One two three four five six seven eight nine ten.",
    ];
    assert_eq!(vertical_trim(inp), out);
}

// Test for any panic from wrong indexing.
#[test]
fn trim_vertically_empty() {
    let inp = &[""];
    let out: &[&str] = &[];
    assert_eq!(vertical_trim(inp), out);

    let inp: &[&str] = &[];
    let out: &[&str] = &[];
    assert_eq!(vertical_trim(inp), out);
}

#[test]
fn trim_horizontally() {
    let inp = &[
        " \t\t * one two three",
        " \t\t * four fix six seven *",
        " \t\t * forty two ",
        " \t\t ** sixty nine",
    ];
    let out: &[&str] = &[" one two three", " four fix six seven *", " forty two ", "* sixty nine"];
    assert_eq!(horizontal_trim(inp).as_deref(), Some(out));

    // Test that we handle empty collection and collection with one item.
    assert_eq!(horizontal_trim(&[]).as_deref(), None);
    assert_eq!(horizontal_trim(&[""]).as_deref(), None);

    // Non-accepted: "\t" will not equal to " "

    let inp = &[
        " \t * one two three",
        "     * four fix six seven *",
        " \t * forty two ",
        " \t ** sixty nine",
    ];
    assert_eq!(horizontal_trim(inp).as_deref(), None);
}

#[test]
fn test_get_prefix() {
    assert_eq!(get_prefix(" \t **"), Some(" \t *"));
    assert_eq!(get_prefix("*"), Some("*"));
    assert_eq!(get_prefix(" \t ^*"), None);
    assert_eq!(get_prefix("   "), None);
}
