// https://github.com/rust-lang/rust/issues/21600
fn call_it<F>(f: F) where F: Fn() { f(); }

struct A;

impl A {
    fn gen(&self) {}
    fn gen_mut(&mut self) {}
}

fn main() {
    let mut x = A;
    call_it(|| {
        call_it(|| x.gen());
        call_it(|| x.gen_mut());
        //~^ ERROR cannot borrow `x` as mutable, as it is a captured variable in a `Fn` closure
        //~| ERROR cannot borrow `x` as mutable, as it is a captured variable in a `Fn` closure
    });
}
