//@ build-pass

trait X<const N: i32> {}

fn hello<T: X<{ fn hello() -> ! { loop {} } 1 }>>() {}

fn main() {}
