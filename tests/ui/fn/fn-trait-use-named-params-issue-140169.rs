fn g(_: fn(a: u8)) {}
fn x(_: impl Fn(u8, vvvv: u8)) {} //~ ERROR expected one of `!`, `(`, `)`, `+`, `,`, `::`, or `<`, found `:`
fn y(_: impl Fn(aaaa: u8, u8)) {} //~ ERROR expected one of `!`, `(`, `)`, `+`, `,`, `::`, or `<`, found `:`
fn z(_: impl Fn(aaaa: u8, vvvv: u8)) {} //~ ERROR expected one of `!`, `(`, `)`, `+`, `,`, `::`, or `<`, found `:`

fn main(){}
