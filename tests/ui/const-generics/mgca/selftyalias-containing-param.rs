#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

struct S<const N: usize>([(); N]);

impl<const N: usize> S<N> {
    fn foo() -> [(); const { let _: Self = loop {}; 1 }] {
    //~^ ERROR generic `Self`
        todo!()
    }
}

fn main() {}
