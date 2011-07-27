// error-pattern: unresolved name: a

mod m1 { }

fn main(args: vec[str]) { log m1::a; }