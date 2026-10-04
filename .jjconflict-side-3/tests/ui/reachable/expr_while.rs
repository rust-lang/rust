#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]

fn foo() {
    while {return} {
        //~^ ERROR unreachable block in `if`
        println!("Hello, world!");
    }
}

fn bar() {
    while {true} {
        return;
    }
    println!("I am not dead.");
}

fn baz() {
    // Here, we cite the `while` loop as dead.
    while {return} {
        //~^ ERROR unreachable block in `if`
        println!("I am dead.");
    }
    println!("I am, too.");
}

fn main() { }
