// compile-flags: -Z parse-only

fn f(+x: isize) {}
//~^ ERROR expected argument name, found `+`
