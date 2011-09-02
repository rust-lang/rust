// error-pattern: unresolved name: a

mod m1 { }

fn main(args: [istr]) { log m1::a; }
