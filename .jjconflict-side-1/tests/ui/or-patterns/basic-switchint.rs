// Test basic or-patterns when the target pattern type will be lowered to
// a `SwitchInt`. This will happen when the target type is an integer.

//@ run-pass

#[derive(Debug, PartialEq)]
enum MatchArm {
    Arm(usize),
    Wild,
}

#[derive(Debug)]
enum Foo {
    One(usize),
    Two(usize, usize),
}

fn test_foo(x: Foo) -> MatchArm {
    match x {
        // normal pattern.
        Foo::One(0) | Foo::One(1) | Foo::One(2) => MatchArm::Arm(0),
        // most simple or-pattern.
        Foo::One(42 | 255) => MatchArm::Arm(1),
        // multiple or-patterns for one structure.
        Foo::Two(42 | 255, 1024 | 2048) => MatchArm::Arm(2),
        // mix of pattern types in one or-pattern (range).
        Foo::One(100 | 110..=120 | 210..=220) => MatchArm::Arm(3),
        // multiple or-patterns with wild.
        Foo::Two(0..=10 | 100..=110, 0 | _) => MatchArm::Arm(4),
        // wild
        _ => MatchArm::Wild,
    }
}

fn main() {
    // `Foo` tests.
    assert_eq!(test_foo(Foo::One(0)), MatchArm::Arm(0));
    assert_eq!(test_foo(Foo::One(42)), MatchArm::Arm(1));
    assert_eq!(test_foo(Foo::One(43)), MatchArm::Wild);
    assert_eq!(test_foo(Foo::One(255)), MatchArm::Arm(1));
    assert_eq!(test_foo(Foo::One(256)), MatchArm::Wild);
    assert_eq!(test_foo(Foo::Two(42, 1023)), MatchArm::Wild);
    assert_eq!(test_foo(Foo::Two(255, 2048)), MatchArm::Arm(2));
    assert_eq!(test_foo(Foo::One(100)), MatchArm::Arm(3));
    assert_eq!(test_foo(Foo::One(115)), MatchArm::Arm(3));
    assert_eq!(test_foo(Foo::One(105)), MatchArm::Wild);
    assert_eq!(test_foo(Foo::One(215)), MatchArm::Arm(3));
    assert_eq!(test_foo(Foo::One(121)), MatchArm::Wild);
    assert_eq!(test_foo(Foo::Two(0, 42)), MatchArm::Arm(4));
    assert_eq!(test_foo(Foo::Two(100, 0)), MatchArm::Arm(4));
    assert_eq!(test_foo(Foo::Two(42, 0)), MatchArm::Wild);
}
