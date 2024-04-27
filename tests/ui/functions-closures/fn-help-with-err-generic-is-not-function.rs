struct Struct<T>(T);
impl Struct<T>
//~^ ERROR cannot find type `T` in this scope
//~| NOTE not found in this scope
//~| HELP you might be missing a type parameter
where
    T: Copy,
    //~^ ERROR cannot find type `T` in this scope
    //~| NOTE not found in this scope
{
    fn method(v: Vec<u8>) { v.len(); }
}

fn main() {}
