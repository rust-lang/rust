/// test the multiline-trim function
extern crate clippy_lints;

use clippy_lints::utils::trim_multiline;

#[test]
fn test_single_line() {
    assert_eq!("", trim_multiline("".into(), false));
    assert_eq!("...", trim_multiline("...".into(), false));
    assert_eq!("...", trim_multiline("    ...".into(), false));
    assert_eq!("...", trim_multiline("\t...".into(), false));
    assert_eq!("...", trim_multiline("\t\t...".into(), false));
}

#[test]
#[cfg_attr(rustfmt, rustfmt_skip)]
fn test_block() {
    assert_eq!("\
if x {
    y
} else {
    z
}", trim_multiline("    if x {
        y
    } else {
        z
    }".into(), false));
    assert_eq!("\
if x {
\ty
} else {
\tz
}", trim_multiline("    if x {
    \ty
    } else {
    \tz
    }".into(), false));
}

#[test]
#[cfg_attr(rustfmt, rustfmt_skip)]
fn test_empty_line() {
    assert_eq!("\
if x {
    y

} else {
    z
}", trim_multiline("    if x {
        y

    } else {
        z
    }".into(), false));
}
