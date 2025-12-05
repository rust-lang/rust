enum E1 {
}

enum E2<T> {
}

enum E3 {
    X
}

enum E4 {
    X,
}

enum E5 {
    A,
    B = 92,
    C {
        a: u32,
        pub b: f64,
    },
    F {},
    D(u32,),
    E(),
}
