// error-pattern:can not pass a dynamically-sized type by value

fn f<T>(+_x: T) {}
fn main() {}
