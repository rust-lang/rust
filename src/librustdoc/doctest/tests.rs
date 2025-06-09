use std::path::PathBuf;

use super::{BuildDocTestBuilder, GlobalTestOptions};

fn make_test(
    test_code: &str,
    crate_name: Option<&str>,
    dont_insert_main: bool,
    opts: &GlobalTestOptions,
    global_crate_attrs: Vec<&str>,
    test_id: Option<&str>,
) -> (String, usize) {
    let mut builder = BuildDocTestBuilder::new(test_code)
        .global_crate_attrs(global_crate_attrs.into_iter().map(|a| a.to_string()).collect());
    if let Some(crate_name) = crate_name {
        builder = builder.crate_name(crate_name);
    }
    if let Some(test_id) = test_id {
        builder = builder.test_id(test_id.to_string());
    }
    let doctest = builder.build(None);
    let (code, line_offset) =
        doctest.generate_unique_doctest(test_code, dont_insert_main, opts, crate_name);
    (code, line_offset)
}

/// Default [`GlobalTestOptions`] for these unit tests.
fn default_global_opts(crate_name: impl Into<String>) -> GlobalTestOptions {
    GlobalTestOptions {
        crate_name: crate_name.into(),
        no_crate_inject: false,
        insert_indent_space: false,
        args_file: PathBuf::new(),
    }
}

#[test]
fn make_test_basic() {
    //basic use: wraps with `fn main`, adds `#![allow(unused)]`
    let opts = default_global_opts("");
    let input = "assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
fn main() {
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len) = make_test(input, None, false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_crate_name_no_use() {
    // If you give a crate name but *don't* use it within the test, it won't bother inserting
    // the `extern crate` statement.
    let opts = default_global_opts("asdf");
    let input = "assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
fn main() {
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len) = make_test(input, Some("asdf"), false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_crate_name() {
    // If you give a crate name and use it within the test, it will insert an `extern crate`
    // statement before `fn main`.
    let opts = default_global_opts("asdf");
    let input = "use asdf::qwop;
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
#[allow(unused_extern_crates)]
extern crate r#asdf;
fn main() {
use asdf::qwop;
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len) = make_test(input, Some("asdf"), false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 3));
}

#[test]
fn make_test_no_crate_inject() {
    // Even if you do use the crate within the test, setting `opts.no_crate_inject` will skip
    // adding it anyway.
    let opts = GlobalTestOptions { no_crate_inject: true, ..default_global_opts("asdf") };
    let input = "use asdf::qwop;
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
fn main() {
use asdf::qwop;
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len) = make_test(input, Some("asdf"), false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_ignore_std() {
    // Even if you include a crate name, and use it in the doctest, we still won't include an
    // `extern crate` statement if the crate is "std" -- that's included already by the
    // compiler!
    let opts = default_global_opts("std");
    let input = "use std::*;
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
fn main() {
use std::*;
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len) = make_test(input, Some("std"), false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_manual_extern_crate() {
    // When you manually include an `extern crate` statement in your doctest, `make_test`
    // assumes you've included one for your own crate too.
    let opts = default_global_opts("asdf");
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
    let (output, len) = make_test(input, Some("asdf"), false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_manual_extern_crate_with_macro_use() {
    let opts = default_global_opts("asdf");
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
    let (output, len) = make_test(input, Some("asdf"), false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_opts_attrs() {
    // If you supplied some doctest attributes with `#![doc(test(attr(...)))]`, it will use
    // those instead of the stock `#![allow(unused)]`.
    let opts = default_global_opts("asdf");
    let input = "use asdf::qwop;
assert_eq!(2+2, 4);";
    let expected = "#![feature(sick_rad)]
#[allow(unused_extern_crates)]
extern crate r#asdf;
fn main() {
use asdf::qwop;
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len) =
        make_test(input, Some("asdf"), false, &opts, vec!["feature(sick_rad)"], None);
    assert_eq!((output, len), (expected, 3));

    let expected = "#![feature(sick_rad)]
#![feature(hella_dope)]
#[allow(unused_extern_crates)]
extern crate r#asdf;
fn main() {
use asdf::qwop;
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len) = make_test(
        input,
        Some("asdf"),
        false,
        &opts,
        vec![
            "feature(sick_rad)",
            // Adding more will also bump the returned line offset.
            "feature(hella_dope)",
        ],
        None,
    );
    assert_eq!((output, len), (expected, 4));
}

#[test]
fn make_test_crate_attrs() {
    // Including inner attributes in your doctest will apply them to the whole "crate", pasting
    // them outside the generated main function.
    let opts = default_global_opts("");
    let input = "#![feature(sick_rad)]
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
#![feature(sick_rad)]

fn main() {
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len) = make_test(input, None, false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_with_main() {
    // Including your own `fn main` wrapper lets the test use it verbatim.
    let opts = default_global_opts("");
    let input = "fn main() {
    assert_eq!(2+2, 4);
}";
    let expected = "#![allow(unused)]
fn main() {
    assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len) = make_test(input, None, false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 1));
}

#[test]
fn make_test_fake_main() {
    // ... but putting it in a comment will still provide a wrapper.
    let opts = default_global_opts("");
    let input = "//Ceci n'est pas une `fn main`
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
fn main() {
//Ceci n'est pas une `fn main`
assert_eq!(2+2, 4);
}"
    .to_string();
    let (output, len) = make_test(input, None, false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_dont_insert_main() {
    // Even with that, if you set `dont_insert_main`, it won't create the `fn main` wrapper.
    let opts = default_global_opts("");
    let input = "//Ceci n'est pas une `fn main`
assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
//Ceci n'est pas une `fn main`
assert_eq!(2+2, 4);"
        .to_string();
    let (output, len) = make_test(input, None, true, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 1));
}

#[test]
fn make_test_issues_21299() {
    let opts = default_global_opts("");

    let input = "// fn main
assert_eq!(2+2, 4);";

    let expected = "#![allow(unused)]
fn main() {
// fn main
assert_eq!(2+2, 4);
}"
    .to_string();

    let (output, len) = make_test(input, None, false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_issues_33731() {
    let opts = default_global_opts("asdf");

    let input = "extern crate hella_qwop;
assert_eq!(asdf::foo, 4);";

    let expected = "#![allow(unused)]
extern crate hella_qwop;
#[allow(unused_extern_crates)]
extern crate r#asdf;
fn main() {
assert_eq!(asdf::foo, 4);
}"
    .to_string();

    let (output, len) = make_test(input, Some("asdf"), false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 3));
}

#[test]
fn make_test_main_in_macro() {
    let opts = default_global_opts("my_crate");
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

    let (output, len) = make_test(input, Some("my_crate"), false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 1));
}

#[test]
fn make_test_returns_result() {
    // creates an inner function and unwraps it
    let opts = default_global_opts("");
    let input = "use std::io;
let mut input = String::new();
io::stdin().read_line(&mut input)?;
Ok::<(), io:Error>(())";
    let expected = "#![allow(unused)]
fn main() { fn _inner() -> core::result::Result<(), impl core::fmt::Debug> {
use std::io;
let mut input = String::new();
io::stdin().read_line(&mut input)?;
Ok::<(), io:Error>(())
} _inner().unwrap() }"
        .to_string();
    let (output, len) = make_test(input, None, false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_named_wrapper() {
    // creates an inner function with a specific name
    let opts = default_global_opts("");
    let input = "assert_eq!(2+2, 4);";
    let expected = "#![allow(unused)]
fn main() { #[allow(non_snake_case)] fn _doctest_main__some_unique_name() {
assert_eq!(2+2, 4);
} _doctest_main__some_unique_name() }"
        .to_string();
    let (output, len) = make_test(input, None, false, &opts, Vec::new(), Some("_some_unique_name"));
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_insert_extra_space() {
    // will insert indent spaces in the code block if `insert_indent_space` is true
    let opts = GlobalTestOptions { insert_indent_space: true, ..default_global_opts("") };
    let input = "use std::*;
assert_eq!(2+2, 4);
eprintln!(\"hello anan\");
";
    let expected = "#![allow(unused)]
fn main() {
    use std::*;
    assert_eq!(2+2, 4);
    eprintln!(\"hello anan\");
}"
    .to_string();
    let (output, len) = make_test(input, None, false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));
}

#[test]
fn make_test_insert_extra_space_fn_main() {
    // if input already has a fn main, it should insert a space before it
    let opts = GlobalTestOptions { insert_indent_space: true, ..default_global_opts("") };
    let input = "use std::*;
fn main() {
    assert_eq!(2+2, 4);
    eprintln!(\"hello anan\");
}";
    let expected = "#![allow(unused)]
use std::*;
fn main() {
    assert_eq!(2+2, 4);
    eprintln!(\"hello anan\");
}"
    .to_string();
    let (output, len) = make_test(input, None, false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 1));
}

#[test]
fn comment_in_attrs() {
    // If there is an inline code comment after attributes, we need to ensure that
    // a backline will be added to prevent generating code "inside" it (and thus generating)
    // invalid code.
    let opts = default_global_opts("");
    let input = "\
#![feature(rustdoc_internals)]
#![allow(internal_features)]
#![doc(rust_logo)]
//! This crate has the Rust(tm) branding on it.";
    let expected = "\
#![allow(unused)]
#![feature(rustdoc_internals)]
#![allow(internal_features)]
#![doc(rust_logo)]
//! This crate has the Rust(tm) branding on it.
fn main() {

}"
    .to_string();
    let (output, len) = make_test(input, None, false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));

    // And same, if there is a `main` function provided by the user, we ensure that it's
    // correctly separated.
    let input = "\
#![feature(rustdoc_internals)]
#![allow(internal_features)]
#![doc(rust_logo)]
//! This crate has the Rust(tm) branding on it.
fn main() {}";
    let expected = "\
#![allow(unused)]
#![feature(rustdoc_internals)]
#![allow(internal_features)]
#![doc(rust_logo)]
//! This crate has the Rust(tm) branding on it.

fn main() {}"
        .to_string();
    let (output, len) = make_test(input, None, false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 1));
}

// This test ensures that the only attributes taken into account when we switch between
// "crate level" content and the rest doesn't include inner attributes span, as it would
// include part of the item and generate broken code.
#[test]
fn inner_attributes() {
    let opts = default_global_opts("");
    let input = r#"
//! A doc comment that applies to the implicit anonymous module of this crate

pub mod outer_module {
    //!! - Still an inner line doc (but with a bang at the beginning)
}
"#;
    let expected = "#![allow(unused)]

//! A doc comment that applies to the implicit anonymous module of this crate


fn main() {
pub mod outer_module {
    //!! - Still an inner line doc (but with a bang at the beginning)
}
}"
    .to_string();
    let (output, len) = make_test(input, None, false, &opts, Vec::new(), None);
    assert_eq!((output, len), (expected, 2));
}
