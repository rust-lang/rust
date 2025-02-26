#![allow(clippy::map_with_unused_argument_over_ranges)]
#![warn(clippy::repeat_vec_with_capacity)]

fn main() {
    {
        vec![Vec::<()>::with_capacity(42); 123];
        //~^ ERROR: repeating `Vec::with_capacity` using `vec![x; n]`, which does not retain capacity
    }

    {
        let n = 123;
        vec![Vec::<()>::with_capacity(42); n];
        //~^ ERROR: repeating `Vec::with_capacity` using `vec![x; n]`, which does not retain capacity
    }

    {
        macro_rules! from_macro {
            ($x:expr) => {
                vec![$x; 123];
            };
        }
        // vec expansion is from another macro, don't lint
        from_macro!(Vec::<()>::with_capacity(42));
    }

    {
        std::iter::repeat(Vec::<()>::with_capacity(42));
        //~^ ERROR: repeating `Vec::with_capacity` using `iter::repeat`, which does not retain capacity
    }

    {
        macro_rules! from_macro {
            ($x:expr) => {
                std::iter::repeat($x)
            };
        }
        from_macro!(Vec::<()>::with_capacity(42));
    }
}
