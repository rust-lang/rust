//@ compile-flags: -Zvalidate-mir -Zinline-mir -Zinline-mir-threshold=500

struct Foo<T>([T; 2]);

impl<T: Default + Copy> Default for Foo<T> {
    fn default(&self) -> Self {
        //~^ ERROR method `default` has a `&self` declaration in the impl, but not in the trait
        Foo([Default::default(); 2])
    }
}

fn field_array() {
    let Foo([a, b]): Foo<i32> = Default::default();
}

fn main() {}
