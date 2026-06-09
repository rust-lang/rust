#![allow(clippy::map_with_unused_argument_over_ranges)]
#![warn(clippy::repeat_vec_with_capacity)]

fn main() {
    {
        vec![Vec::<()>::with_capacity(42); 123];
        //~^ repeat_vec_with_capacity
    }

    {
        let n = 123;
        vec![Vec::<()>::with_capacity(42); n];
        //~^ repeat_vec_with_capacity
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
        //~^ repeat_vec_with_capacity
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

#[clippy::msrv = "1.27.0"]
fn msrv_check() {
    std::iter::repeat(Vec::<()>::with_capacity(42));
}
