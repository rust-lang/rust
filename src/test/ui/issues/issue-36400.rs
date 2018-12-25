fn f(x: &mut u32) {}

fn main() {
    let x = Box::new(3);
    f(&mut *x); //~ ERROR cannot borrow immutable
}
