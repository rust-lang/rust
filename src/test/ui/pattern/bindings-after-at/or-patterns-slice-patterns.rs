// Test bindings-after-at with or-patterns and slice-patterns

// run-pass

#![feature(bindings_after_at)]
#![feature(or_patterns)]

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

fn test(foo: &[Option<Test>]) -> MatchArm {
    match foo {
        bar @ [Some(Test::Foo), .., Some(Test::Foo | Test::Qux)] => {
            assert_eq!(bar, foo);

            MatchArm::Arm(0)
        },
        [.., bar @ Some(Test::Bar | Test::Qux), _] => {
            assert!(bar == &Some(Test::Bar) || bar == &Some(Test::Qux));

            MatchArm::Arm(1)
        },
        _ => MatchArm::Wild,
    }
}

fn main() {
    let foo = vec![Some(Test::Foo), Some(Test::Bar), Some(Test::Baz), Some(Test::Qux)];
    assert_eq!(test(&foo), MatchArm::Arm(0));
    assert_eq!(test(&foo[..3]), MatchArm::Arm(1));
    assert_eq!(test(&foo[1..2]), MatchArm::Wild);
}
