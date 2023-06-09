// run-pass

fn f() {
    let a: Box<_> = Box::new(1);
    let b: &isize = &*a;
    println!("{}", b);
}

pub fn main() {
    f();
}
