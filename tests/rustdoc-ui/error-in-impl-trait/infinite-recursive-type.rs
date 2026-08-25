fn f() -> impl Sized {
    enum E {
        //~^ ERROR: recursive type
        V(E),
    }

    unimplemented!()
}
