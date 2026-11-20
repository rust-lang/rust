use std::marker;

fn send<T:Send + std::fmt::Debug>(ch: Chan<T>, data: T) {
    println!("{:?}", ch);
    println!("{:?}", data);
    panic!();
}

#[derive(Debug)]
struct Chan<T>(isize, marker::PhantomData<T>);

// Tests that "log(debug, message);" is flagged as using
// message after the send deinitializes it
fn test00_start(ch: Chan<Box<isize>>, message: Box<isize>, _count: Box<isize>) {
    send(ch, message);
    println!("{}", message); //~ ERROR borrow of moved value: `message`
}

fn main() { panic!(); }
