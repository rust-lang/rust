#![feature(plugin)]
#![plugin(clippy)]

#[deny(collapsible_if)]
fn main() {
    let x = "hello";
    let y = "world";
    if x == "hello" { //~ERROR this if statement can be collapsed
        if y == "world" {
            println!("Hello world!");
        }
    }

    if x == "hello" || x == "world" { //~ERROR this if statement can be collapsed
        if y == "world" || y == "hello" {
            println!("Hello world!");
        }
    }

    // Works because any if with an else statement cannot be collapsed.
    if x == "hello" {
        if y == "world" {
            println!("Hello world!");
        }
    } else {
        println!("Not Hello world");
    }

    if x == "hello" {
        if y == "world" {
            println!("Hello world!");
        } else {
            println!("Hello something else");
        }
    }

    if x == "hello" {
        print!("Hello ");
        if y == "world" {
            println!("world!")
        }
    }
}
