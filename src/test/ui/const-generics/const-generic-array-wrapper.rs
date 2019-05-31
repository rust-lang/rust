// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct Foo<T, const N: usize>([T; {N}]);

impl<T, const N: usize> Foo<T, {N}> {
    fn foo(&self) -> usize {
        {N}
    }
}

fn main() {
    let foo = Foo([0u32; 21]);
    assert_eq!(foo.0, [0u32; 21]);
    assert_eq!(foo.foo(), 21);
}
