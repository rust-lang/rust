// Test or-patterns with slice-patterns

// run-pass

#![feature(or_patterns)]

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
        [.., Some(Test::Foo | Test::Qux)] => MatchArm::Arm(0),
        [Some(Test::Foo), .., Some(Test::Bar | Test::Baz)] => MatchArm::Arm(1),
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

    assert_eq!(test(&foo), MatchArm::Arm(0));
    assert_eq!(test(&foo[..3]), MatchArm::Arm(1));
    assert_eq!(test(&foo[1..3]), MatchArm::Arm(2));
    assert_eq!(test(&foo[4..]), MatchArm::Wild);
}
