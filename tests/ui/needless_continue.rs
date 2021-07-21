#![warn(clippy::needless_continue)]

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
        continue; // should lint here
    }
}

fn simple_loop2() {
    loop {
        println!("bleh");
        continue; // should lint here
    }
}

#[rustfmt::skip]
fn simple_loop3() {
    loop {
        continue // should lint here
    }
}

#[rustfmt::skip]
fn simple_loop4() {
    loop {
        println!("bleh");
        continue // should lint here
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
                    continue 'inner; // should lint here
                }
                println!("bar-4");

                update_condition();
                if condition() {
                    continue; // should lint here
                } else {
                    println!("bar-5");
                }
                println!("bar-6");
            }
        }
    }
}
