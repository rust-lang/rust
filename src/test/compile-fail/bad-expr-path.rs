// error-pattern: unresolved name: m1::a

mod m1 { }

fn main(args: [str]) { log(debug, m1::a); }
