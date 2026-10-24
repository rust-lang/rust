#![warn(clippy::if_not_else)]

fn foo() -> bool {
    unimplemented!()
}
fn bla() -> bool {
    unimplemented!()
}

fn main() {
    if !bla() {
        //~^ if_not_else

        println!("Bugs");
    } else {
        println!("Bunny");
    }
    if 4 != 5 {
        //~^ if_not_else

        println!("Bugs");
    } else {
        println!("Bunny");
    }
    if !foo() {
        println!("Foo");
    } else if !bla() {
        println!("Bugs");
    } else {
        println!("Bunny");
    }

    if !(foo() && bla()) {
        //~^ if_not_else
        #[cfg(not(debug_assertions))]
        println!("not debug");
        #[cfg(debug_assertions)]
        println!("debug");
        if foo() {
            println!("foo");
        } else if bla() {
            println!("bla");
        } else {
            println!("both false");
        }
    } else {
        println!("both true");
    }
}

fn with_comments() {
    if !foo() {
        //~^ if_not_else
        /* foo is false */
        println!("foo is false");
    } else {
        println!("foo"); /* foo */
    }

    if !bla() {
        //~^ if_not_else
        // bla is false
        println!("bla");
    } else {
        println!("bla"); // bla
    }
}

fn with_annotations() {
    #[cfg(debug_assertions)]
    if !foo() {
        //~^ if_not_else
        /* foo is false */
        println!("foo is false");
    } else {
        println!("foo"); /* foo */
    }
}

fn issue15924() {
    let x = 0;
    if !matches!(x, 0..10) {
        //~^ if_not_else
        println!(":)");
    } else {
        println!(":(");
    }

    if dbg!(x) != 1 {
        //~^ if_not_else
        println!(":)");
    } else {
        println!(":(");
    }
}

mod issue17373 {
    macro_rules! ht_is_packed {
        ($ht:expr) => {
            ($ht).u & 4 != 0
        };
    }
    struct HashTable {
        u: u32,
    }
    impl HashTable {
        fn check(&mut self) {
            if ht_is_packed!(self) {
                println!("Packed");
            } else {
                println!("Not packed");
            }
        }
    }

    macro_rules! is_active {
        ($x:expr) => {
            ($x).state != 3
        };
    }
    struct Widget {
        state: u32,
    }
    impl Widget {
        fn check(&self) {
            if is_active!(self) {
                println!("active");
            } else {
                println!("inactive");
            }
        }
    }

    macro_rules! not_ready {
        ($x:expr) => {
            !($x).ready
        };
    }
    struct Task {
        ready: bool,
    }
    impl Task {
        fn check(&self) {
            if not_ready!(self) {
                println!("waiting");
            } else {
                println!("go");
            }
        }
    }

    macro_rules! wrap_if {
        ($cond:expr, $t:block, $e:block) => {
            if $cond $t else $e
        };
    }
    fn wrap_if_macro() {
        let x = 0;
        fn a() {}
        fn b() {}
        wrap_if!(x != 0, { a() }, { b() });
    }
}
