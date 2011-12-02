// error-pattern: unresolved name: m1::a

mod m1 { }

fn main(args: [str]) { log m1::a; }
