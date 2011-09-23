

fn leaky<T>(t: T) { }

fn main() { let x = ~10; leaky::<~int>(x); }
