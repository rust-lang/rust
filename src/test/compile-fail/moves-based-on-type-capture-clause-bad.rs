use std::task;

fn main() {
    let x = ~"Hello world!";
    task::spawn(proc() {
        println!("{}", x);
    });
    println!("{}", x); //~ ERROR use of moved value
}
