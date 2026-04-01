// Test bindings-after-at with box-patterns

//@ run-pass

#![feature(box_patterns)]

#[derive(Debug, PartialEq)]
enum MatchArm {
    Arm(usize),
    Wild,
}

fn test(x: Option<Box<i32>>) -> MatchArm {
    match x {
        ref bar @ Some(box n) if n > 0 => {
            // bar is a &Option<Box<i32>>
            assert_eq!(bar, &x);

            MatchArm::Arm(0)
        },
        Some(ref bar @ box n) if n < 0 => {
            // bar is a &Box<i32> here
            assert_eq!(**bar, n);

            MatchArm::Arm(1)
        },
        _ => MatchArm::Wild,
    }
}

fn main() {
    assert_eq!(test(Some(Box::new(2))), MatchArm::Arm(0));
    assert_eq!(test(Some(Box::new(-1))), MatchArm::Arm(1));
    assert_eq!(test(Some(Box::new(0))), MatchArm::Wild);
}
