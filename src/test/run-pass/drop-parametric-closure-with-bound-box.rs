

fn f<T>(i: @uint, t: T) { }

fn main() { let x = f::<char>(@0xdeafbeefu, _); }
