import core::*;

use std;
import char;

#[test]
fn test_is_lowercase() {
    assert char::is_lowercase('a');
    assert char::is_lowercase('ö');
    assert char::is_lowercase('ß');
    assert !char::is_lowercase('Ü');
    assert !char::is_lowercase('P');
}

#[test]
fn test_is_uppercase() {
    assert !char::is_uppercase('h');
    assert !char::is_uppercase('ä');
    assert !char::is_uppercase('ß');
    assert char::is_uppercase('Ö');
    assert char::is_uppercase('T');
}

#[test]
fn test_is_whitespace() {
    assert char::is_whitespace(' ');
    assert char::is_whitespace('\u2007');
    assert char::is_whitespace('\t');
    assert char::is_whitespace('\n');

    assert !char::is_whitespace('a');
    assert !char::is_whitespace('_');
    assert !char::is_whitespace('\u0000');
}

#[test]
fn test_to_digit() {
    assert (char::to_digit('0') == 0u8);
    assert (char::to_digit('1') == 1u8);
    assert (char::to_digit('2') == 2u8);
    assert (char::to_digit('9') == 9u8);
    assert (char::to_digit('a') == 10u8);
    assert (char::to_digit('A') == 10u8);
    assert (char::to_digit('b') == 11u8);
    assert (char::to_digit('B') == 11u8);
    assert (char::to_digit('z') == 35u8);
    assert (char::to_digit('Z') == 35u8);
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_to_digit_fail_1() {
    char::to_digit(' ');
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_to_digit_fail_2() {
    char::to_digit('$');
}

#[test]
fn test_to_lowercase() {
    assert (char::to_lowercase('H') == 'h');
    assert (char::to_lowercase('e') == 'e');
    //assert (char::to_lowercase('Ö') == 'ö');
    assert (char::to_lowercase('ß') == 'ß');
}

#[test]
fn test_to_uppercase() {
    assert (char::to_uppercase('l') == 'L');
    assert (char::to_uppercase('Q') == 'Q');
    //assert (char::to_uppercase('ü') == 'Ü');
    assert (char::to_uppercase('ß') == 'ß');
}
