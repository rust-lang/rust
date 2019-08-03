fn avg<T = i32>(_: T) {}
//~^ ERROR defaults for type parameters are only allowed

struct S<T>(T);
impl<T = i32> S<T> {}
//~^ ERROR defaults for type parameters are only allowed

fn main() {}
