// Test bindings-after-at with or-patterns and box-patterns

//@ run-pass

#![feature(box_patterns)]

#[derive(Debug, PartialEq)]
enum MatchArm {
    Arm(usize),
    Wild,
}

#[derive(Debug, PartialEq)]
enum Test {
    Foo,
    Bar,
    Baz,
    Qux,
}

fn test(foo: Option<Box<Test>>) -> MatchArm {
    match foo {
        ref bar @ Some(box Test::Foo | box Test::Bar) => {
            assert_eq!(bar, &foo);

            MatchArm::Arm(0)
        },
        Some(ref bar @ box Test::Baz | ref bar @ box Test::Qux) => {
            assert!(**bar == Test::Baz || **bar == Test::Qux);

            MatchArm::Arm(1)
        },
        _ => MatchArm::Wild,
    }
}

fn main() {
    assert_eq!(test(Some(Box::new(Test::Foo))), MatchArm::Arm(0));
    assert_eq!(test(Some(Box::new(Test::Bar))), MatchArm::Arm(0));
    assert_eq!(test(Some(Box::new(Test::Baz))), MatchArm::Arm(1));
    assert_eq!(test(Some(Box::new(Test::Qux))), MatchArm::Arm(1));
    assert_eq!(test(None), MatchArm::Wild);
}
