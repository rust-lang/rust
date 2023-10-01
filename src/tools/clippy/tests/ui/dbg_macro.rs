//@no-rustfix

#![warn(clippy::dbg_macro)]

fn foo(n: u32) -> u32 {
    if let Some(n) = dbg!(n.checked_sub(4)) { n } else { n }
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    //~| NOTE: `-D clippy::dbg-macro` implied by `-D warnings`
}
fn bar(_: ()) {}

fn factorial(n: u32) -> u32 {
    if dbg!(n <= 1) {
        //~^ ERROR: the `dbg!` macro is intended as a debugging tool
        dbg!(1)
        //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    } else {
        dbg!(n * factorial(n - 1))
        //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    }
}

fn main() {
    dbg!(42);
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    dbg!(dbg!(dbg!(42)));
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    foo(3) + dbg!(factorial(4));
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    dbg!(1, 2, dbg!(3, 4));
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    dbg!(1, 2, 3, 4, 5);
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
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
        };
    }

    dbg!();
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    #[allow(clippy::let_unit_value)]
    let _ = dbg!();
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    bar(dbg!());
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    foo!(dbg!());
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    foo2!(foo!(dbg!()));
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
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
        //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    });
}

#[test]
pub fn issue8481() {
    dbg!(1);
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
}

#[cfg(test)]
fn foo2() {
    dbg!(1);
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
}

#[cfg(test)]
mod mod1 {
    fn func() {
        dbg!(1);
        //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    }
}
