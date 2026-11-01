trait Deserializer<'a> {}

trait Deserializable {
    fn deserialize_token<'a, D: Deserializer<'a>>(_: D, _: &'a str) -> Self;
}

impl<'a, T: Deserializable> Deserializable for &'a str {
    //~^ ERROR type parameter `T` is not constrained
    fn deserialize_token<D: Deserializer<'a>>(_x: D, _y: &'a str) -> &'a str {
        //~^ ERROR mismatched types
        //~| ERROR do not match the trait declaration
    }
}

fn main() {}
