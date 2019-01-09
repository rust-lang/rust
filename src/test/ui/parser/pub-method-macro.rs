// Issue #18317

mod bleh {
    macro_rules! defn {
        ($n:ident) => (
            fn $n (&self) -> i32 {
                println!("{}", stringify!($n));
                1
            }
        )
    }

    #[derive(Copy, Clone)]
    pub struct S;

    impl S {
        pub defn!(f); //~ ERROR can't qualify macro invocation with `pub`
        //~^ HELP try adjusting the macro to put `pub` inside the invocation
    }
}

fn main() {}
