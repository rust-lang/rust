//@check-pass

#![feature(const_trait_impl, comptime)]

struct Bar<T>(T);

const trait Trait {
    fn method(&self) {}
}

#[comptime]
impl<T: const Trait> Bar<T> {
    fn boo(&self) {
        self.0.method()
    }
}

const impl Trait for () {}

const _: () = {
    Bar(()).boo();
};

fn main() {}
