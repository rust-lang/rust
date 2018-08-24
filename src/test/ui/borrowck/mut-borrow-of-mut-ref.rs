// Suggest not mutably borrowing a mutable reference

fn main() {
    f(&mut 0)
}

fn f(b: &mut i32) {
    g(&mut b) //~ ERROR cannot borrow
}

fn g(_: &mut i32) {}
