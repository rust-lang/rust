trait Foo {
    fn test() -> impl Sized + use<Self>;
    //~^ ERROR `use<...>` precise capturing syntax is currently not allowed in return-position
}

fn main() {}
