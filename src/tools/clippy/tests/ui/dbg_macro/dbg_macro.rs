#![allow(
    clippy::no_effect,
    clippy::uninlined_format_args,
    clippy::unit_arg,
    clippy::unnecessary_operation
)]
#![warn(clippy::dbg_macro)]

fn foo(n: u32) -> u32 {
    if let Some(n) = dbg!(n.checked_sub(4)) { n } else { n }
    //~^ dbg_macro
}
fn bar(_: ()) {}

fn factorial(n: u32) -> u32 {
    if dbg!(n <= 1) {
        //~^ dbg_macro

        dbg!(1)
        //~^ dbg_macro
    } else {
        dbg!(n * factorial(n - 1))
        //~^ dbg_macro
    }
}

fn main() {
    dbg!(42);
    //~^ dbg_macro

    foo(3) + dbg!(factorial(4));
    //~^ dbg_macro

    dbg!(1, 2, 3, 4, 5);
    //~^ dbg_macro
}

fn issue9914() {
    macro_rules! foo {
        ($x:expr) => {
            $x;
        };
    }
    macro_rules! foo2 {
        ($x:expr) => {
            $x;
        };
    }
    macro_rules! expand_to_dbg {
        () => {
            dbg!();
            //~^ dbg_macro
        };
    }

    dbg!();
    //~^ dbg_macro

    #[allow(clippy::let_unit_value)]
    let _ = dbg!();
    //~^ dbg_macro

    bar(dbg!());
    //~^ dbg_macro

    foo!(dbg!());
    //~^ dbg_macro

    foo2!(foo!(dbg!()));
    //~^ dbg_macro

    expand_to_dbg!();
}

mod issue7274 {
    trait Thing<'b> {
        fn foo(&self);
    }

    macro_rules! define_thing {
        ($thing:ident, $body:expr) => {
            impl<'a> Thing<'a> for $thing {
                fn foo<'b>(&self) {
                    $body
                }
            }
        };
    }

    struct MyThing;
    define_thing!(MyThing, {
        dbg!(2);
        //~^ dbg_macro
    });
}

#[test]
pub fn issue8481() {
    dbg!(1);
    //~^ dbg_macro
}

#[cfg(test)]
fn foo2() {
    dbg!(1);
    //~^ dbg_macro
}

#[cfg(test)]
mod mod1 {
    fn func() {
        dbg!(1);
        //~^ dbg_macro
    }
}

mod issue12131 {
    fn dbg_in_print(s: &str) {
        println!("dbg: {:?}", dbg!(s));
        //~^ dbg_macro

        print!("{}", dbg!(s));
        //~^ dbg_macro
    }
}
