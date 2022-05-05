#![allow(unused, clippy::print_literal, clippy::write_literal)]
#![warn(clippy::print_in_format_impl)]
use std::fmt::{Debug, Display, Error, Formatter};

macro_rules! indirect {
    () => {{ println!() }};
}

macro_rules! nested {
    ($($tt:tt)*) => {
        $($tt)*
    };
}

struct Foo;
impl Debug for Foo {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        static WORKS_WITH_NESTED_ITEMS: bool = true;

        print!("{}", 1);
        println!("{}", 2);
        eprint!("{}", 3);
        eprintln!("{}", 4);
        nested! {
            println!("nested");
        };

        write!(f, "{}", 5);
        writeln!(f, "{}", 6);
        indirect!();

        Ok(())
    }
}

impl Display for Foo {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        print!("Display");
        write!(f, "Display");

        Ok(())
    }
}

struct UnnamedFormatter;
impl Debug for UnnamedFormatter {
    fn fmt(&self, _: &mut Formatter) -> Result<(), Error> {
        println!("UnnamedFormatter");
        Ok(())
    }
}

fn main() {
    print!("outside fmt");
    println!("outside fmt");
    eprint!("outside fmt");
    eprintln!("outside fmt");
}
