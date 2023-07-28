#![allow(unused, clippy::print_literal, clippy::write_literal)]
#![warn(clippy::print_in_format_impl)]
use std::fmt::{Debug, Display, Error, Formatter};
//@no-rustfix
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
        //~^ ERROR: use of `print!` in `Debug` impl
        //~| NOTE: `-D clippy::print-in-format-impl` implied by `-D warnings`
        println!("{}", 2);
        //~^ ERROR: use of `println!` in `Debug` impl
        eprint!("{}", 3);
        //~^ ERROR: use of `eprint!` in `Debug` impl
        eprintln!("{}", 4);
        //~^ ERROR: use of `eprintln!` in `Debug` impl
        nested! {
            println!("nested");
            //~^ ERROR: use of `println!` in `Debug` impl
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
        //~^ ERROR: use of `print!` in `Display` impl
        write!(f, "Display");

        Ok(())
    }
}

struct UnnamedFormatter;
impl Debug for UnnamedFormatter {
    fn fmt(&self, _: &mut Formatter) -> Result<(), Error> {
        println!("UnnamedFormatter");
        //~^ ERROR: use of `println!` in `Debug` impl
        Ok(())
    }
}

fn main() {
    print!("outside fmt");
    println!("outside fmt");
    eprint!("outside fmt");
    eprintln!("outside fmt");
}
