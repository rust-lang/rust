use self::Self as Foo; //~ ERROR unresolved import `self::Self`

pub fn main() {
    let Self = 5;
    //~^ ERROR cannot find unit struct, unit variant or constant `Self`

    match 15 {
        Self => (),
        //~^ ERROR cannot find unit struct, unit variant or constant `Self`
        Foo { x: Self } => (),
        //~^ ERROR cannot find unit struct, unit variant or constant `Self`
    }
}
