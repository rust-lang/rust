// check-pass

fn f() -> impl Sized {
    enum E {
        V(E),
    }

    unimplemented!()
}
