//@ revisions: type_infer const_infer int_infer float_infer
//@ compile-flags: -Znext-solver
//@ check-pass

#[cfg(type_infer)]
mod test {
    pub fn run() {
        let mut values = Vec::new();

        values.push(Default::default());

        // Resolving the element type makes fulfillment revisit the stalled `Default` obligation
        values.push(0u8);
    }
}

#[cfg(const_infer)]
mod test {
    trait Marker<const N: usize> {}

    // Keep `N` ambiguous here because a single impl would let trait solving infer it
    impl Marker<0> for () {}
    impl Marker<1> for () {}

    fn value<const N: usize>() -> [(); N]
    where
        (): Marker<N>,
    {
        [(); N]
    }

    pub fn run() {
        let value = value::<_>();

        // Resolve the const after the `Marker` obligation has stalled
        let _: [(); 1] = value;
    }
}

#[cfg(int_infer)]
mod test {
    trait Marker {
        fn marker(self);
    }

    impl Marker for u8 {
        fn marker(self) {}
    }

    impl Marker for u16 {
        fn marker(self) {}
    }

    pub fn run() {
        let value = 0;

        // Keep the obligation directly on the integer inference variable
        value.marker();

        let _: u8 = value;
    }
}

#[cfg(float_infer)]
mod test {
    trait Marker {
        fn marker(self);
    }

    impl Marker for f32 {
        fn marker(self) {}
    }

    impl Marker for f64 {
        fn marker(self) {}
    }

    pub fn run() {
        let value = 0.0;

        // Keep the obligation directly on the float inference variable
        value.marker();

        let _: f32 = value;
    }
}

fn main() {
    test::run();
}
