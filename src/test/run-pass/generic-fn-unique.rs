
fn f<copy T>(x: ~T) -> ~T { ret x; }

fn main() { let x = f(~3); log(debug, *x); }
