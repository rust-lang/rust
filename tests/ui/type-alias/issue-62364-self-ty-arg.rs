struct Struct<P1> {
    field: P1,
}

type Alias<'a> = Struct<&'a Self>;
//~^ ERROR cannot find type `Self` [E0411]

fn main() {}
