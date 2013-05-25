use std::task;

pub fn main() {
    let x = ~"Hello world!";
    do task::spawn {
        println(x);
    }
}
