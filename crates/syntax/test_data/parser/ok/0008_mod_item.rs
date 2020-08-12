mod a;

mod b {
}

mod c {
    fn foo() {
    }
    struct S {}
}

mod d {
    #![attr]
    mod e;
    mod f {
    }
}