//@ run-pass
// https://github.com/rust-lang/rust/issues/111229

pub struct Foo(std::cell::UnsafeCell<usize>);
pub struct Bar([u8; 0]);

pub fn foo(f: &Bar) {
    unsafe {
        let f = std::mem::transmute::<&Bar, &Foo>(f);
        //~^ WARNING transmuting from a type without interior mutability to a type with interior mutability
        //~| HELP `Foo` has interior mutability
        *(f.0.get()) += 1;
    }
}

fn main() {}
