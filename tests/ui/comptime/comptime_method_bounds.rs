//@check-pass

#![feature(rustc_attrs, const_trait_impl)]

struct Bar<T>(T);

const trait Trait {
    fn method(&self) {}
}

#[rustc_comptime]
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
