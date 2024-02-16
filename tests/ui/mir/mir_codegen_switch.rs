//@ run-pass
enum Abc {
    A(#[allow(dead_code)] u8),
    B(#[allow(dead_code)] i8),
    C,
    D,
}

fn foo(x: Abc) -> i32 {
    match x {
        Abc::C => 3,
        Abc::D => 4,
        Abc::B(_) => 2,
        Abc::A(_) => 1,
    }
}

fn foo2(x: Abc) -> bool {
    match x {
        Abc::D => true,
        _ => false
    }
}

fn main() {
    assert_eq!(1, foo(Abc::A(42)));
    assert_eq!(2, foo(Abc::B(-100)));
    assert_eq!(3, foo(Abc::C));
    assert_eq!(4, foo(Abc::D));

    assert_eq!(false, foo2(Abc::A(1)));
    assert_eq!(false, foo2(Abc::B(2)));
    assert_eq!(false, foo2(Abc::C));
    assert_eq!(true, foo2(Abc::D));
}
