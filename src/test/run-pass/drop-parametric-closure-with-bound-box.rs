

fn f<T>(i: @uint, t: T) { }

fn main() { let x = bind f::<char>(@0xdeafbeefu, _); }
