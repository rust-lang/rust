#![feature(plugin)]
#![plugin(clippy)]
#![deny(nonminimal_bool, logic_bug)]

#[allow(unused, many_single_char_names)]
fn main() {
    let a: bool = unimplemented!();
    let b: bool = unimplemented!();
    let c: bool = unimplemented!();
    let d: bool = unimplemented!();
    let e: bool = unimplemented!();
    let _ = a && b || a; //~ ERROR this boolean expression contains a logic bug
    //~| HELP this expression can be optimized out
    //~| HELP it would look like the following
    //~| SUGGESTION let _ = a;
    let _ = !(a && b);
    let _ = !true; //~ ERROR this boolean expression can be simplified
    //~| HELP try
    //~| SUGGESTION let _ = false;
    let _ = !false; //~ ERROR this boolean expression can be simplified
    //~| HELP try
    //~| SUGGESTION let _ = true;
    let _ = !!a; //~ ERROR this boolean expression can be simplified
    //~| HELP try
    //~| SUGGESTION let _ = a;

    let _ = false && a; //~ ERROR this boolean expression contains a logic bug
    //~| HELP this expression can be optimized out
    //~| HELP it would look like the following
    //~| SUGGESTION let _ = false;

    let _ = false || a; //~ ERROR this boolean expression can be simplified
    //~| HELP try
    //~| SUGGESTION let _ = a;

    // don't lint on cfgs
    let _ = cfg!(you_shall_not_not_pass) && a;

    let _ = a || !b || !c || !d || !e;

    let _ = !(a && b || c);

    let _ = !(!a && b); //~ ERROR this boolean expression can be simplified
    //~| HELP try
    //~| SUGGESTION let _ = !b || a;
}

#[allow(unused, many_single_char_names)]
fn equality_stuff() {
    let a: i32 = unimplemented!();
    let b: i32 = unimplemented!();
    let c: i32 = unimplemented!();
    let d: i32 = unimplemented!();
    let e: i32 = unimplemented!();
    let _ = a == b && a != b;
    //~^ ERROR this boolean expression contains a logic bug
    //~| HELP this expression can be optimized out
    //~| HELP it would look like the following
    //~| SUGGESTION let _ = false;
    let _ = a == b && c == 5 && a == b;
    //~^ ERROR this boolean expression can be simplified
    //~| HELP try
    //~| SUGGESTION let _ = a == b && c == 5;
    //~| HELP try
    //~| SUGGESTION let _ = !(c != 5 || a != b);
    let _ = a == b && c == 5 && b == a;
    //~^ ERROR this boolean expression can be simplified
    //~| HELP try
    //~| SUGGESTION let _ = a == b && c == 5;
    //~| HELP try
    //~| SUGGESTION let _ = !(c != 5 || a != b);
    let _ = a < b && a >= b;
    //~^ ERROR this boolean expression contains a logic bug
    //~| HELP this expression can be optimized out
    //~| HELP it would look like the following
    //~| SUGGESTION let _ = false;
    let _ = a > b && a <= b;
    //~^ ERROR this boolean expression contains a logic bug
    //~| HELP this expression can be optimized out
    //~| HELP it would look like the following
    //~| SUGGESTION let _ = false;
    let _ = a > b && a == b;

    let _ = a != b || !(a != b || c == d);
    //~^ ERROR this boolean expression can be simplified
    //~| HELP try
    //~| SUGGESTION let _ = c != d || a != b;
    //~| HELP try
    //~| SUGGESTION let _ = !(a == b && c == d);
}
