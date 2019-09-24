// build-pass (FIXME(62277): could be check-pass?)

#![warn(unused_parens)]

macro_rules! the_worship_the_heart_lifts_above {
    ( @as_expr, $e:expr) => { $e };
    ( @generate_fn, $name:tt) => {
        #[allow(dead_code)] fn the_moth_for_the_star<'a>() -> Option<&'a str> {
            Some(the_worship_the_heart_lifts_above!( @as_expr, $name ))
        }
    };
    ( $name:ident ) => { the_worship_the_heart_lifts_above!( @generate_fn, (stringify!($name))); }
    // ↑ Notably, this does 𝘯𝘰𝘵 warn: we're declining to lint unused parens in
    // function/method arguments inside of nested macros because of situations
    // like those reported in Issue #47775
}

macro_rules! and_the_heavens_reject_not {
    () => {
        // ↓ But let's test that we still lint for unused parens around
        // function args inside of simple, one-deep macros.
        #[allow(dead_code)] fn the_night_for_the_morrow() -> Option<isize> { Some((2)) }
        //~^ WARN unnecessary parentheses around function argument
    }
}

the_worship_the_heart_lifts_above!(rah);
and_the_heavens_reject_not!();

fn main() {}
