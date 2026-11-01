// Test bindings-after-at with or-patterns

//@ run-pass


#[derive(Debug, PartialEq)]
enum MatchArm {
    Arm(usize),
    Wild,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Test {
    Foo,
    Bar,
    Baz,
    Qux,
}

fn test(foo: Option<Test>) -> MatchArm {
    match foo {
        bar @ Some(Test::Foo | Test::Bar) => {
            assert!(bar == Some(Test::Foo) || bar == Some(Test::Bar));

            MatchArm::Arm(0)
        },
        Some(_) => MatchArm::Arm(1),
        _ => MatchArm::Wild,
    }
}

fn main() {
    assert_eq!(test(Some(Test::Foo)), MatchArm::Arm(0));
    assert_eq!(test(Some(Test::Bar)), MatchArm::Arm(0));
    assert_eq!(test(Some(Test::Baz)), MatchArm::Arm(1));
    assert_eq!(test(Some(Test::Qux)), MatchArm::Arm(1));
    assert_eq!(test(None), MatchArm::Wild);
}
