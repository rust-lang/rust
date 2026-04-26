#![feature(fn_delegation)]

mod test_ice {
    fn a() {}

    reuse a as b { //~ ERROR: this function takes 0 arguments but 1 argument was supplied
        let closure = || {
            fn foo<'a, 'b, T: Clone, const N: usize, U: Clone>(_t: &'a T, _u: &'b U) {}

            reuse foo::<String, 1, String> as bar;
            bar(&"".to_string(), &"".to_string());
        };

        closure();
    }
}

mod test_2 {
    mod to_reuse {
        pub fn zero_args() -> i32 {
            15
        }
    }

    reuse to_reuse::zero_args { self }
    //~^ ERROR: this function takes 0 arguments but 1 argument was supplied
}

fn main() {}
