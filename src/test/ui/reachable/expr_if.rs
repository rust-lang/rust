#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]

fn foo() {
    if {return} { //~ ERROR unreachable block in `if` expression
        println!("Hello, world!");
    }
}

fn bar() {
    if {true} {
        return;
    }
    println!("I am not dead.");
}

fn baz() {
    if {true} {
        return;
    } else {
        return;
    }
    // As the next action to be taken after the if arms, we should
    // report the `println!` as unreachable:
    println!("But I am.");
    //~^ ERROR unreachable statement
}

fn main() { }
