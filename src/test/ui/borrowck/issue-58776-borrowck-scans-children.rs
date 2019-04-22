// ignore-compare-mode-nll

// revisions: migrate nll

#![cfg_attr(nll, feature(nll))]

fn main() {
    let mut greeting = "Hello world!".to_string();
    let res = (|| (|| &greeting)())();

    greeting = "DEALLOCATED".to_string();
    //[migrate]~^ ERROR cannot assign
    //[nll]~^^ ERROR cannot assign
    drop(greeting);
    //[migrate]~^ ERROR cannot move
    //[nll]~^^ ERROR cannot move

    println!("thread result: {:?}", res);
}
