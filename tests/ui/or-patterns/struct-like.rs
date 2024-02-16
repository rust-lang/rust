//@ run-pass

#[derive(Debug)]
enum Other {
    One,
    Two,
    Three,
}

#[derive(Debug)]
enum Test {
    Foo { first: usize, second: usize },
    Bar { other: Option<Other> },
    Baz,
}

fn test(x: Option<Test>) -> bool {
    match x {
        Some(
            Test::Foo { first: 1024 | 2048, second: 2048 | 4096 }
            | Test::Bar { other: Some(Other::One | Other::Two) },
        ) => true,
        // wild case
        Some(_) => false,
        // empty case
        None => false,
    }
}

fn main() {
    assert!(test(Some(Test::Foo { first: 1024, second: 4096 })));
    assert!(!test(Some(Test::Foo { first: 2048, second: 8192 })));
    assert!(!test(Some(Test::Foo { first: 42, second: 2048 })));
    assert!(test(Some(Test::Bar { other: Some(Other::One) })));
    assert!(test(Some(Test::Bar { other: Some(Other::Two) })));
    assert!(!test(Some(Test::Bar { other: Some(Other::Three) })));
    assert!(!test(Some(Test::Bar { other: None })));
    assert!(!test(Some(Test::Baz)));
    assert!(!test(None));
}
