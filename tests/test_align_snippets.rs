extern crate clippy_lints;

use clippy_lints::utils::align_snippets;

#[test]
fn test_align_snippets_single_line() {
    assert_eq!("", align_snippets(&[""]));
    assert_eq!("...", align_snippets(&["..."]));
}

#[test]
#[cfg_attr(rustfmt, rustfmt_skip)]
fn test_align_snippets_multiline() {
    let expected = "\
if condition() {
    do_something();
    do_another_thing();
    yet_another_thing();
    {
        and_then_an_indented_block();
    }
        and_then_something_the_user_indented();"; // expected

    let input = &[
"\
if condition() {
    do_something();",
"       do_another_thing();",
"            yet_another_thing();
            {
                and_then_an_indented_block();
            }
                and_then_something_the_user_indented();",
    ]; // input

    let got = align_snippets(input);
    assert_eq!(expected, got);

}

#[test]
#[cfg_attr(rustfmt, rustfmt_skip)]
fn test_align_snippets_multiline_with_empty_lines() {
    let expected = "\
if condition() {
    do_something();
    do_another_thing();
    yet_another_thing();
    {

        and_then_an_indented_block();
    }

        and_then_something_the_user_indented();"; // expected

    let input = &[
"\
if condition() {
    do_something();",
"       do_another_thing();",
"            yet_another_thing();
            {

                and_then_an_indented_block();
            }

                and_then_something_the_user_indented();",
    ]; // input

    let got = align_snippets(input);
    println!("Input: {}\nExpected: {}\nGot: {}", input.join("\n"), &expected, &got);
    assert_eq!(expected, got);
}

