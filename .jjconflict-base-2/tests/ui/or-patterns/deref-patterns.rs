// Test or-patterns with deref patterns

//@ run-pass

#![feature(deref_patterns)]

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

fn test(x: Option<Box<Test>>) -> MatchArm {
    match x {
        Some(Test::Foo | Test::Bar) => MatchArm::Arm(0),
        Some(Test::Baz) => MatchArm::Arm(1),
        Some(_) => MatchArm::Arm(2),
        _ => MatchArm::Wild,
    }
}

fn main() {
    assert_eq!(test(Some(Box::new(Test::Foo))), MatchArm::Arm(0));
    assert_eq!(test(Some(Box::new(Test::Bar))), MatchArm::Arm(0));
    assert_eq!(test(Some(Box::new(Test::Baz))), MatchArm::Arm(1));
    assert_eq!(test(Some(Box::new(Test::Qux))), MatchArm::Arm(2));
    assert_eq!(test(None), MatchArm::Wild);
}
