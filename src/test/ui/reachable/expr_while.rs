#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]

fn foo() {
    while {return} {
        println!("Hello, world!");
        //~^ ERROR unreachable
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
        println!("I am dead.");
        //~^ ERROR unreachable
    }
    println!("I am, too.");
    //~^ ERROR unreachable
}

fn main() { }
