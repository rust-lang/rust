enum E<W: ?Sized> {
    // parameter
    VA(W),
    //~^ ERROR the size for values of type

    // slice / str
    VF{x: str},
    //~^ ERROR the size for values of type
}


fn main() { }
