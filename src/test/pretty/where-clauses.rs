// pp-exact

fn f<'a, 'b, T>(t: T) -> isize where T: 'a, 'a: 'b, T: Eq { 0 }

fn main() {}
