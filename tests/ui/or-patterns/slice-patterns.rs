// Test or-patterns with slice-patterns

//@ run-pass

#[derive(Debug, PartialEq)]
enum MatchArm {
    Arm(usize),
    Wild,
}

#[derive(Debug)]
enum Test {
    Foo,
    Bar,
    Baz,
    Qux,
}

fn test(foo: &[Option<Test>]) -> MatchArm {
    match foo {
        [.., Some(Test::Qux | Test::Foo)] => MatchArm::Arm(0),
        [Some(Test::Foo), .., Some(Test::Baz | Test::Bar)] => MatchArm::Arm(1),
        [.., Some(Test::Bar | Test::Baz), _] => MatchArm::Arm(2),
        _ => MatchArm::Wild,
    }
}

fn main() {
    let foo = vec![
        Some(Test::Foo),
        Some(Test::Bar),
        Some(Test::Baz),
        Some(Test::Qux),
    ];

    // path 1a
    assert_eq!(test(&foo), MatchArm::Arm(0));
    // path 1b
    assert_eq!(test(&[Some(Test::Bar), Some(Test::Foo)]), MatchArm::Arm(0));
    // path 2a
    assert_eq!(test(&foo[..3]), MatchArm::Arm(1));
    // path 2b
    assert_eq!(test(&[Some(Test::Foo), Some(Test::Foo), Some(Test::Bar)]), MatchArm::Arm(1));
    // path 3a
    assert_eq!(test(&foo[1..3]), MatchArm::Arm(2));
    // path 3b
    assert_eq!(test(&[Some(Test::Bar), Some(Test::Baz), Some(Test::Baz), Some(Test::Bar)]),
        MatchArm::Arm(2));
    // path 4
    assert_eq!(test(&foo[4..]), MatchArm::Wild);
}
