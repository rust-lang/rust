// error-pattern: unresolved name: a

mod m1 { }

fn main(args: [str]) { log m1::a; }