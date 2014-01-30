use std::task;

pub fn main() {
    let x = ~"Hello world!";
    task::spawn(proc() {
        println!("{}", x);
    });
}
