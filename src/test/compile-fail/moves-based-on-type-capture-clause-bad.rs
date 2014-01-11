use std::task;

fn main() {
    let x = ~"Hello world!";
    do task::spawn {
        println!("{}", x);
    }
    println!("{}", x); //~ ERROR use of moved value
}
