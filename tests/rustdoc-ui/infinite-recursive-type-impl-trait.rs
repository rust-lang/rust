fn f() -> impl Sized {
    enum E { //~ ERROR
        V(E),
    }
    unimplemented!()
}
