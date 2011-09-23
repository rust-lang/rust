// xfail-test

fn f<T>(x: ~T) -> ~T { ret x; }

fn main() { let x = f(~3); log *x; }
