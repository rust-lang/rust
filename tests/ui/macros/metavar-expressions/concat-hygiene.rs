#![feature(macro_metavar_expr_concat)]

macro_rules! join {
    ($lhs:ident, $rhs:ident) => {
        ${concat($lhs, $rhs)}
        //~^ ERROR cannot find value `abcdef` in this scope
    };
}

fn main() {
    let abcdef = 1;
    let _another = join!(abc, def);
}
