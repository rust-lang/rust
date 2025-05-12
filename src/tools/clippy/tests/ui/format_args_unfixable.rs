#![warn(clippy::format_in_format_args, clippy::to_string_in_format_args)]
#![allow(unused)]
#![allow(clippy::assertions_on_constants, clippy::eq_op, clippy::uninlined_format_args)]

use std::io::{Error, Write, stdout};
use std::ops::Deref;
use std::panic::Location;

macro_rules! my_macro {
    () => {
        // here be dragons, do not enter (or lint)
        println!("error: {}", format!("something failed at {}", Location::caller()));
    };
}

macro_rules! my_other_macro {
    () => {
        format!("something failed at {}", Location::caller())
    };
}

fn main() {
    let error = Error::other("bad thing");
    let x = 'x';

    println!("error: {}", format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    println!("{}: {}", error, format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    println!("{:?}: {}", error, format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    println!("{{}}: {}", format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    println!(r#"error: "{}""#, format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    println!("error: {}", format!(r#"something failed at "{}""#, Location::caller()));
    //~^ format_in_format_args

    println!("error: {}", format!("something failed at {} {0}", Location::caller()));
    //~^ format_in_format_args

    let _ = format!("error: {}", format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    let _ = write!(
        //~^ format_in_format_args
        stdout(),
        "error: {}",
        format!("something failed at {}", Location::caller())
    );
    let _ = writeln!(
        //~^ format_in_format_args
        stdout(),
        "error: {}",
        format!("something failed at {}", Location::caller())
    );
    print!("error: {}", format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    eprint!("error: {}", format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    eprintln!("error: {}", format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    let _ = format_args!("error: {}", format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    assert!(true, "error: {}", format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    assert_eq!(0, 0, "error: {}", format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    assert_ne!(0, 0, "error: {}", format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    panic!("error: {}", format!("something failed at {}", Location::caller()));
    //~^ format_in_format_args

    // negative tests
    println!("error: {}", format_args!("something failed at {}", Location::caller()));
    println!("error: {:>70}", format!("something failed at {}", Location::caller()));
    println!("error: {} {0}", format!("something failed at {}", Location::caller()));
    println!("{} and again {0}", format!("hi {}", x));
    my_macro!();
    println!("error: {}", my_other_macro!());
}

macro_rules! _internal {
    ($($args:tt)*) => {
        println!("{}", format_args!($($args)*))
    };
}

macro_rules! my_println2 {
   ($target:expr, $($args:tt)+) => {{
       if $target {
           _internal!($($args)+)
       }
    }};
}

macro_rules! my_println2_args {
    ($target:expr, $($args:tt)+) => {{
       if $target {
           _internal!("foo: {}", format_args!($($args)+))
       }
    }};
}

fn test2() {
    let error = Error::other("bad thing");

    // None of these should be linted without the config change
    my_println2!(true, "error: {}", format!("something failed at {}", Location::caller()));
    my_println2!(
        true,
        "{}: {}",
        error,
        format!("something failed at {}", Location::caller())
    );

    my_println2_args!(true, "error: {}", format!("something failed at {}", Location::caller()));
    my_println2_args!(
        true,
        "{}: {}",
        error,
        format!("something failed at {}", Location::caller())
    );
}

#[clippy::format_args]
macro_rules! usr_println {
    ($target:expr, $($args:tt)*) => {{
        if $target {
            println!($($args)*)
        }
    }};
}

fn user_format() {
    let error = Error::other("bad thing");
    let x = 'x';

    usr_println!(true, "error: {}", format!("boom at {}", Location::caller()));
    //~^ format_in_format_args

    usr_println!(true, "{}: {}", error, format!("boom at {}", Location::caller()));
    //~^ format_in_format_args

    usr_println!(true, "{:?}: {}", error, format!("boom at {}", Location::caller()));
    //~^ format_in_format_args

    usr_println!(true, "{{}}: {}", format!("boom at {}", Location::caller()));
    //~^ format_in_format_args

    usr_println!(true, r#"error: "{}""#, format!("boom at {}", Location::caller()));
    //~^ format_in_format_args

    usr_println!(true, "error: {}", format!(r#"boom at "{}""#, Location::caller()));
    //~^ format_in_format_args

    usr_println!(true, "error: {}", format!("boom at {} {0}", Location::caller()));
    //~^ format_in_format_args
}
