//@ run-pass



fn main() {
    #[allow(dead_code)]
    enum MyEnum {
        A(String),
        B { f: String },
        C,
    }
    // ref binding to non-copy value and or-pattern
    let (MyEnum::A(ref x) | MyEnum::B { f: ref x }) = (MyEnum::B { f: String::new() }) else {
        panic!();
    };
    assert_eq!(x, "");

    // nested let-else
    let mut x = 1;
    loop {
        let 4 = x else {
            let 3 = x else {
                x += 1;
                continue;
            };
            break;
        };
        panic!();
    }
    assert_eq!(x, 3);

    // else return
    let Some(1) = Some(2) else { return };
    panic!();
}
