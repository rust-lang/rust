struct Struct<T>(T);
impl Struct<T>
//~^ ERROR cannot find type `T`
//~| NOTE not found
//~| HELP you might be missing a type parameter
where
    T: Copy,
    //~^ ERROR cannot find type `T`
    //~| NOTE not found
{
    fn method(v: Vec<u8>) { v.len(); }
}

fn main() {}
