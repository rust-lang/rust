#![allow(unused)]

fn avg<T=i32>(_: T) {}
//~^ ERROR defaults for type parameters are only allowed
//~| WARN this was previously accepted

struct S<T>(T);
impl<T=i32> S<T> {}
//~^ ERROR defaults for type parameters are only allowed
//~| WARN this was previously accepted

fn main() {}
