//@ run-pass

fn f<T>(x: Box<T>) -> Box<T> { return x; }

pub fn main() {
    let x = f(Box::new(3));
    println!("{}", *x);
}
