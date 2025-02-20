#![warn(clippy::needless_continue)]
#![allow(clippy::uninlined_format_args)]

macro_rules! zero {
    ($x:expr) => {
        $x == 0
    };
}

macro_rules! nonzero {
    ($x:expr) => {
        !zero!($x)
    };
}

#[allow(clippy::nonminimal_bool)]
fn main() {
    let mut i = 1;
    while i < 10 {
        i += 1;

        if i % 2 == 0 && i % 3 == 0 {
            println!("{}", i);
            println!("{}", i + 1);
            if i % 5 == 0 {
                println!("{}", i + 2);
            }
            let i = 0;
            println!("bar {} ", i);
        } else {
            //~^ needless_continue

            continue;
        }

        println!("bleh");
        {
            println!("blah");
        }

        // some comments that also should ideally be included in the
        // output of the lint suggestion if possible.
        if !(!(i == 2) || !(i == 5)) {
            println!("lama");
        }

        if (zero!(i % 2) || nonzero!(i % 5)) && i % 3 != 0 {
            //~^ needless_continue

            continue;
        } else {
            println!("Blabber");
            println!("Jabber");
        }

        println!("bleh");
    }
}

fn simple_loop() {
    loop {
        continue;
        //~^ needless_continue
    }
}

fn simple_loop2() {
    loop {
        println!("bleh");
        continue;
        //~^ needless_continue
    }
}

#[rustfmt::skip]
fn simple_loop3() {
    loop {
        continue
        //~^ needless_continue

    }
}

#[rustfmt::skip]
fn simple_loop4() {
    loop {
        println!("bleh");
        continue
        //~^ needless_continue

    }
}

fn simple_loop5() {
    loop {
        println!("bleh");
        { continue }
        //~^ needless_continue
    }
}

mod issue_2329 {
    fn condition() -> bool {
        unimplemented!()
    }
    fn update_condition() {}

    // only the outer loop has a label
    fn foo() {
        'outer: loop {
            println!("Entry");
            while condition() {
                update_condition();
                if condition() {
                    println!("foo-1");
                } else {
                    continue 'outer; // should not lint here
                }
                println!("foo-2");

                update_condition();
                if condition() {
                    continue 'outer; // should not lint here
                } else {
                    println!("foo-3");
                }
                println!("foo-4");
            }
        }
    }

    // both loops have labels
    fn bar() {
        'outer: loop {
            println!("Entry");
            'inner: while condition() {
                update_condition();
                if condition() {
                    println!("bar-1");
                } else {
                    continue 'outer; // should not lint here
                }
                println!("bar-2");

                update_condition();
                if condition() {
                    println!("bar-3");
                } else {
                    //~^ needless_continue

                    continue 'inner;
                }
                println!("bar-4");

                update_condition();
                if condition() {
                    //~^ needless_continue

                    continue;
                } else {
                    println!("bar-5");
                }
                println!("bar-6");
            }
        }
    }
}

fn issue_13641() {
    'a: while std::hint::black_box(true) {
        #[allow(clippy::never_loop)]
        loop {
            continue 'a;
        }
    }

    #[allow(clippy::never_loop)]
    while std::hint::black_box(true) {
        'b: loop {
            continue 'b;
            //~^ needless_continue
        }
    }
}

mod issue_4077 {
    fn main() {
        'outer: loop {
            'inner: loop {
                do_something();
                if some_expr() {
                    println!("bar-7");
                    continue 'outer;
                } else if !some_expr() {
                    println!("bar-8");
                    continue 'inner;
                    //~^ needless_continue
                } else {
                    println!("bar-9");
                    continue 'inner;
                    //~^ needless_continue
                }
            }
        }

        for _ in 0..10 {
            match "foo".parse::<i32>() {
                Ok(_) => do_something(),
                Err(_) => {
                    println!("bar-10");
                    continue;
                    //~^ needless_continue
                },
            }
        }

        loop {
            if true {
            } else {
                //~^ needless_continue
                // redundant `else`
                continue; // redundant `continue`
            }
        }

        loop {
            if some_expr() {
                //~^ needless_continue
                continue;
            } else {
                do_something();
            }
        }
    }

    // The contents of these functions are irrelevant, the purpose of this file is
    // shown in main.

    fn do_something() {
        std::process::exit(0);
    }

    fn some_expr() -> bool {
        true
    }
}
