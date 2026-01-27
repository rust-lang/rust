// While mentioning `Self` in the method signature of dyn compatible traits is generally forbidden
// due to type erasure, we can make an exception for const projections from `Self` where the trait
// is the principal trait or a supertrait thereof. That's sound because we force users to specify
// all associated consts in the trait object type, so the projections are all normalizable.
//
// Check that we can define & use dyn compatible traits that reference `Self` const projections.

// This is a run-pass test to ensure that codegen can actually deal with such method instances
// (e.g., const projections normalize flawlessly to something concrete, symbols get mangled
// properly, the vtable is fine) and simply to ensure that the generated code "received" the
// correct values from the type assoc consts).
//@ run-pass

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait {
    #[type_const]
    const N: usize;

    fn process(&self, _: [u8; Self::N]) -> [u8; Self::N];
}

impl Trait for u8 {
    #[type_const]
    const N: usize = 2;

    fn process(&self, [x, y]: [u8; Self::N]) -> [u8; Self::N] {
        [self * x, self + y]
    }
}

impl<const N: usize> Trait for [u8; N] {
    #[type_const]
    const N: usize = N;

    fn process(&self, other: [u8; Self::N]) -> [u8; Self::N] {
        let mut result = [0; _];
        for i in 0..Self::N {
            result[i] = self[i] + other[i];
        }
        result
    }
}

fn main() {
    let ops: [Box<dyn Trait<N = 2>>; _] = [Box::new(3), Box::new([1, 1])];

    let mut data = [16, 32];

    for op in ops {
        data = op.process(data);
    }

    assert_eq!(data, [49, 36]);
}
