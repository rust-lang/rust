// check-pass

fn f() -> impl Sized {
    // rustdoc doesn't care that this is infinitely sized
    enum E {
        V(E),
    }
    unimplemented!()
}
