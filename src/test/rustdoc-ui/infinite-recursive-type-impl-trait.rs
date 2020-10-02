fn f() -> impl Sized {
    enum E {
    //~^ ERROR recursive type `f::E` has infinite size
        V(E),
    }
    unimplemented!()
}
