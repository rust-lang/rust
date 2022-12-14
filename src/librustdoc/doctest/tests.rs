use super::{make_test, GlobalTestOptions};
use rustc_span::edition::DEFAULT_EDITION;

#[test]
fn make_test_basic() {
    //basic use: wraps with `fn main`, adds `#![allow(unused)]`
    let opts = GlobalTestOptions::default();
    let input = "assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
fn main() {
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len, _) = make_test(input, None, false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_crate_name_no_use() {
    // If you give a crate name but *don't* use it within the test, it won't bother inserting
    // the `extern crate` statement.
    let opts = GlobalTestOptions::default();
    let input = "assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
fn main() {
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len, _) = make_test(input, Some("asdf"), false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_crate_name() {
    // If you give a crate name and use it within the test, it will insert an `extern crate`
    // statement before `fn main`.
    let opts = GlobalTestOptions::default();
    let input = "use asdf::qwop;
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
extern crate r#asdf;
fn main() {
use asdf::qwop;
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len, _) = make_test(input, Some("asdf"), false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 3));
}

#[test]
fn make_test_no_crate_inject() {
    // Even if you do use the crate within the test, setting `opts.no_crate_inject` will skip
    // adding it anyway.
    let opts = GlobalTestOptions { no_crate_inject: true, attrs: vec![] };
    let input = "use asdf::qwop;
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
fn main() {
use asdf::qwop;
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len, _) = make_test(input, Some("asdf"), false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_ignore_std() {
    // Even if you include a crate name, and use it in the doctest, we still won't include an
    // `extern crate` statement if the crate is "std" -- that's included already by the
    // compiler!
    let opts = GlobalTestOptions::default();
    let input = "use std::*;
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
fn main() {
use std::*;
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len, _) = make_test(input, Some("std"), false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_manual_extern_crate() {
    // When you manually include an `extern crate` statement in your doctest, `make_test`
    // assumes you've included one for your own crate too.
    let opts = GlobalTestOptions::default();
    let input = "extern crate asdf;
use asdf::qwop;
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
extern crate asdf;
fn main() {
use asdf::qwop;
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len, _) = make_test(input, Some("asdf"), false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_manual_extern_crate_with_macro_use() {
    let opts = GlobalTestOptions::default();
    let input = "#[macro_use] extern crate asdf;
use asdf::qwop;
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
#[macro_use] extern crate asdf;
fn main() {
use asdf::qwop;
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len, _) = make_test(input, Some("asdf"), false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_opts_attrs() {
    // If you supplied some doctest attributes with `#![doc(test(attr(...)))]`, it will use
    // those instead of the stock `#![allow(unused)]`.
    let mut opts = GlobalTestOptions::default();
    opts.attrs.push("feature(sick_rad)".to_string());
    let input = "use asdf::qwop;
assert_eq!(2+2, 4);";
    let expected = "#![feature(sick_rad)]
extern crate r#asdf;
fn main() {
use asdf::qwop;
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len, _) = make_test(input, Some("asdf"), false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 3));

    // Adding more will also bump the returned line offset.
    opts.attrs.push("feature(hella_dope)".to_string());
    let expected = "#![feature(sick_rad)]
#![feature(hella_dope)]
extern crate r#asdf;
fn main() {
use asdf::qwop;
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len, _) = make_test(input, Some("asdf"), false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 4));
}

#[test]
fn make_test_crate_attrs() {
    // Including inner attributes in your doctest will apply them to the whole "crate", pasting
    // them outside the generated main function.
    let opts = GlobalTestOptions::default();
    let input = "#![feature(sick_rad)]
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
#![feature(sick_rad)]
fn main() {
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len, _) = make_test(input, None, false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_with_main() {
    // Including your own `fn main` wrapper lets the test use it verbatim.
    let opts = GlobalTestOptions::default();
    let input = "fn main() {
    assert_eq!(2+2, 4);
}";
    let expected = "#![allow(unused)]
fn main() {
    assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len, _) = make_test(input, None, false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 1));
}

#[test]
fn make_test_fake_main() {
    // ... but putting it in a comment will still provide a wrapper.
    let opts = GlobalTestOptions::default();
    let input = "//Ceci n'est pas une `fn main`
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
//Ceci n'est pas une `fn main`
fn main() {
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len, _) = make_test(input, None, false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_dont_insert_main() {
    // Even with that, if you set `dont_insert_main`, it won't create the `fn main` wrapper.
    let opts = GlobalTestOptions::default();
    let input = "//Ceci n'est pas une `fn main`
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
//Ceci n'est pas une `fn main`
assert_eq!(2+2, 4);"
        .to_string();
    let (output, len, _) = make_test(input, None, true, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 1));
}

#[test]
fn make_test_issues_21299_33731() {
    let opts = GlobalTestOptions::default();

    let input = "// fn main
assert_eq!(2+2, 4);";

    let expected = "#![allow(unused)]
// fn main
fn main() {
assert_eq!(2+2, 4);
}"
    .to_string();

    let (output, len, _) = make_test(input, None, false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 2));

    let input = "extern crate hella_qwop;
assert_eq!(asdf::foo, 4);";

    let expected = "#![allow(unused)]
extern crate hella_qwop;
extern crate r#asdf;
fn main() {
assert_eq!(asdf::foo, 4);
}"
    .to_string();

    let (output, len, _) = make_test(input, Some("asdf"), false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 3));
}

#[test]
fn make_test_main_in_macro() {
    let opts = GlobalTestOptions::default();
    let input = "#[macro_use] extern crate my_crate;
test_wrapper! {
    fn main() {}
}";
    let expected = "#![allow(unused)]
#[macro_use] extern crate my_crate;
test_wrapper! {
    fn main() {}
}"
    .to_string();

    let (output, len, _) = make_test(input, Some("my_crate"), false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 1));
}

#[test]
fn make_test_returns_result() {
    // creates an inner function and unwraps it
    let opts = GlobalTestOptions::default();
    let input = "use std::io;
let mut input = String::new();
io::stdin().read_line(&mut input)?;
Ok::<(), io:Error>(())";
    let expected = "#![allow(unused)]
fn main() { fn _inner() -> Result<(), impl core::fmt::Debug> {
use std::io;
let mut input = String::new();
io::stdin().read_line(&mut input)?;
Ok::<(), io:Error>(())
} _inner().unwrap() }"
        .to_string();
    let (output, len, _) = make_test(input, None, false, &opts, DEFAULT_EDITION, None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_named_wrapper() {
    // creates an inner function with a specific name
    let opts = GlobalTestOptions::default();
    let input = "assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
fn main() { #[allow(non_snake_case)] fn _doctest_main__some_unique_name() {
assert_eq!(2+2, 4);
} _doctest_main__some_unique_name() }"
        .to_string();
    let (output, len, _) =
        make_test(input, None, false, &opts, DEFAULT_EDITION, Some("_some_unique_name"));
    assert_eq!((output, len), (expected, 2));
}
