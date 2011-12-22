// error-pattern: unresolved name: m1::a

mod m1 { }

fn main(args: [str]) { log_full(core::debug, m1::a); }
