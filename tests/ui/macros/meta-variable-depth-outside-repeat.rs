#![feature(macro_metavar_expr)]

macro_rules! metavar {
    ( $i:expr ) => {
        ${length(0)}
        //~^ ERROR meta-variable expression `length` with depth parameter must be called inside of a macro repetition
    };
}

const _: i32 = metavar!(0);

fn main() {}
