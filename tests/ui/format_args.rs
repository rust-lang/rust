#![warn(clippy::to_string_in_format_args)]
#![allow(unused)]
#![allow(
    clippy::assertions_on_constants,
    clippy::double_parens,
    clippy::eq_op,
    clippy::print_literal,
    clippy::uninlined_format_args
)]

use std::io::{Write, stdout};
use std::ops::Deref;
use std::panic::Location;

struct Somewhere;

#[allow(clippy::to_string_trait_impl)]
impl ToString for Somewhere {
    fn to_string(&self) -> String {
        String::from("somewhere")
    }
}

struct X(u32);

impl Deref for X {
    type Target = u32;

    fn deref(&self) -> &u32 {
        &self.0
    }
}

struct Y<'a>(&'a X);

impl<'a> Deref for Y<'a> {
    type Target = &'a X;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct Z(u32);

impl Deref for Z {
    type Target = u32;

    fn deref(&self) -> &u32 {
        &self.0
    }
}

impl std::fmt::Display for Z {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Z")
    }
}

macro_rules! my_macro {
    () => {
        // here be dragons, do not enter (or lint)
        println!("error: something failed at {}", Location::caller().to_string());
    };
}

macro_rules! my_other_macro {
    () => {
        Location::caller().to_string()
    };
}

fn main() {
    let x = &X(1);
    let x_ref = &x;

    let _ = format!("error: something failed at {}", Location::caller().to_string());
    //~^ to_string_in_format_args
    let _ = write!(
        stdout(),
        "error: something failed at {}",
        Location::caller().to_string(),
        //~^ to_string_in_format_args
    );
    let _ = writeln!(
        stdout(),
        "error: something failed at {}",
        Location::caller().to_string(),
        //~^ to_string_in_format_args
    );
    print!("error: something failed at {}", Location::caller().to_string());
    //~^ to_string_in_format_args
    println!("error: something failed at {}", Location::caller().to_string());
    //~^ to_string_in_format_args
    eprint!("error: something failed at {}", Location::caller().to_string());
    //~^ to_string_in_format_args
    eprintln!("error: something failed at {}", Location::caller().to_string());
    //~^ to_string_in_format_args
    let _ = format_args!("error: something failed at {}", Location::caller().to_string());
    //~^ to_string_in_format_args
    assert!(true, "error: something failed at {}", Location::caller().to_string());
    //~^ to_string_in_format_args
    assert_eq!(0, 0, "error: something failed at {}", Location::caller().to_string());
    //~^ to_string_in_format_args
    assert_ne!(0, 0, "error: something failed at {}", Location::caller().to_string());
    //~^ to_string_in_format_args
    panic!("error: something failed at {}", Location::caller().to_string());
    //~^ to_string_in_format_args
    println!("{}", X(1).to_string());
    //~^ to_string_in_format_args
    println!("{}", Y(&X(1)).to_string());
    //~^ to_string_in_format_args
    println!("{}", Z(1).to_string());
    //~^ to_string_in_format_args
    println!("{}", x.to_string());
    //~^ to_string_in_format_args
    println!("{}", x_ref.to_string());
    //~^ to_string_in_format_args
    // https://github.com/rust-lang/rust-clippy/issues/7903
    println!("{foo}{bar}", foo = "foo".to_string(), bar = "bar");
    //~^ to_string_in_format_args
    println!("{foo}{bar}", foo = "foo", bar = "bar".to_string());
    //~^ to_string_in_format_args
    println!("{foo}{bar}", bar = "bar".to_string(), foo = "foo");
    //~^ to_string_in_format_args
    println!("{foo}{bar}", bar = "bar", foo = "foo".to_string());
    //~^ to_string_in_format_args
    println!("{}", my_other_macro!().to_string());
    //~^ to_string_in_format_args

    // negative tests
    println!("error: something failed at {}", Somewhere.to_string());
    // The next two tests are negative because caching the string might be faster than calling `<X as
    // Display>::fmt` twice.
    println!("{} and again {0}", x.to_string());
    println!("{foo}{foo}", foo = "foo".to_string());
    my_macro!();
    println!("error: something failed at {}", my_other_macro!());
    // https://github.com/rust-lang/rust-clippy/issues/7903
    println!("{foo}{foo:?}", foo = "foo".to_string());
    print!("{}", (Location::caller().to_string()));
    //~^ to_string_in_format_args
    print!("{}", ((Location::caller()).to_string()));
    //~^ to_string_in_format_args
}

fn issue8643(vendor_id: usize, product_id: usize, name: &str) {
    println!(
        "{:<9}  {:<10}  {}",
        format!("0x{:x}", vendor_id),
        format!("0x{:x}", product_id),
        name
    );
}

// https://github.com/rust-lang/rust-clippy/issues/8855
mod issue_8855 {
    #![allow(dead_code)]

    struct A {}

    impl std::fmt::Display for A {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "test")
        }
    }

    fn main() {
        let a = A {};
        let b = A {};

        let x = format!("{} {}", a, b.to_string());
        //~^ to_string_in_format_args
        dbg!(x);

        let x = format!("{:>6} {:>6}", a, b.to_string());
        dbg!(x);
    }
}

// https://github.com/rust-lang/rust-clippy/issues/9256
mod issue_9256 {
    #![allow(dead_code)]

    fn print_substring(original: &str) {
        assert!(original.len() > 10);
        println!("{}", original[..10].to_string());
        //~^ to_string_in_format_args
    }

    fn main() {
        print_substring("Hello, world!");
    }
}

mod issue14952 {
    use std::path::Path;
    struct Foo(Path);
    impl std::fmt::Debug for Foo {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", &self.0)
        }
    }
}
