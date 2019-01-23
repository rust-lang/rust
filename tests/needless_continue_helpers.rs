// Tests for the various helper functions used by the needless_continue
// lint that don't belong in utils.

use clippy_lints::needless_continue::{erode_block, erode_from_back, erode_from_front};

#[test]
#[rustfmt::skip]
fn test_erode_from_back() {
    let input = "\
{
    let x = 5;
    let y = format!(\"{}\", 42);
}";

    let expected = "\
{
    let x = 5;
    let y = format!(\"{}\", 42);";

    let got = erode_from_back(input);
    assert_eq!(expected, got);
}

#[test]
#[rustfmt::skip]
fn test_erode_from_back_no_brace() {
    let input = "\
let x = 5;
let y = something();
";
    let expected = "";
    let got = erode_from_back(input);
    assert_eq!(expected, got);
}

#[test]
#[rustfmt::skip]
fn test_erode_from_front() {
    let input = "
        {
            something();
            inside_a_block();
        }
    ";
    let expected =
"            something();
            inside_a_block();
        }
    ";
    let got = erode_from_front(input);
    println!("input: {}\nexpected:\n{}\ngot:\n{}", input, expected, got);
    assert_eq!(expected, got);
}

#[test]
#[rustfmt::skip]
fn test_erode_from_front_no_brace() {
    let input = "
            something();
            inside_a_block();
    ";
    let expected =
"something();
            inside_a_block();
    ";
    let got = erode_from_front(input);
    println!("input: {}\nexpected:\n{}\ngot:\n{}", input, expected, got);
    assert_eq!(expected, got);
}

#[test]
#[rustfmt::skip]
fn test_erode_block() {

    let input = "
        {
            something();
            inside_a_block();
        }
    ";
    let expected =
"            something();
            inside_a_block();";
    let got = erode_block(input);
    println!("input: {}\nexpected:\n{}\ngot:\n{}", input, expected, got);
    assert_eq!(expected, got);
}
