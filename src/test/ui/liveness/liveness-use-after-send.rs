use std::marker;

fn send<T:Send + std::fmt::Debug>(ch: _chan<T>, data: T) {
    println!("{:?}", ch);
    println!("{:?}", data);
    panic!();
}

#[derive(Debug)]
struct _chan<T>(isize, marker::PhantomData<T>);

// Tests that "log(debug, message);" is flagged as using
// message after the send deinitializes it
fn test00_start(ch: _chan<Box<isize>>, message: Box<isize>, _count: Box<isize>) {
    send(ch, message);
    println!("{}", message); //~ ERROR use of moved value: `message`
}

fn main() { panic!(); }
