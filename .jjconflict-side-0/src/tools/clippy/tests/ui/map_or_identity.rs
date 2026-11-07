#![warn(clippy::map_or_identity)]

mod issue15801 {

    fn foo(opt: Option<i32>, default: i32) -> i32 {
        opt.map_or(default, |o| o)
        //~^ map_or_identity
    }

    fn bar(res: Result<i32, &str>, default: i32) -> i32 {
        res.map_or(default, |o| o)
        //~^ map_or_identity
    }

    fn with_deref(opt: &Option<i32>, default: i32) -> i32 {
        opt.map_or(default, |o| o)
        //~^ map_or_identity
    }
}

mod macros {
    macro_rules! mac {
        ($e:expr) => {{ $e }};
    }

    fn option_with_macro(opt: Option<i32>) -> i32 {
        opt.map_or(mac!(42), |x| x)
        //~^ map_or_identity
    }

    fn result_with_macro(res: Result<i32, &str>) -> i32 {
        res.map_or(mac!(42), |x| x)
        //~^ map_or_identity
    }

    fn option_with_macro_receiver(opt: Option<i32>) -> i32 {
        mac!(opt).map_or(42, |x| x)
        //~^ map_or_identity
    }

    fn result_with_macro_receiver(res: Result<i32, &str>) -> i32 {
        mac!(res).map_or(42, |x| x)
        //~^ map_or_identity
    }

    // These should not lint because the method call comes from a macro expansion
    macro_rules! map_or_mac {
        ($e:expr, $d:expr, $f:expr) => {
            $e.map_or($d, $f)
        };
    }

    fn option_with_macro_call(opt: Option<i32>) -> i32 {
        map_or_mac!(opt, 42, |x| x)
    }

    fn result_with_macro_call(res: Result<i32, &str>) -> i32 {
        map_or_mac!(res, 42, |x| x)
    }
}

fn main() {
    // test code goes here
}
