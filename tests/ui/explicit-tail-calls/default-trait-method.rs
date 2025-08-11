// A regression test for <https://github.com/rust-lang/rust/issues/144985>.
// Previously, using `become` in a default trait method would lead to an ICE
// in a path determining whether the method in question is marked as `#[track_caller]`.
//
//@ run-pass

#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

trait Trait {
    fn bar(&self) -> usize {
        123
    }

    fn foo(&self) -> usize {
        become self.bar();
    }
}

struct Struct;

impl Trait for Struct {}

struct OtherStruct;

impl Trait for OtherStruct {
    #[track_caller]
    fn bar(&self) -> usize {
        456
    }
}

fn main() {
    assert_eq!(Struct.foo(), 123);
    assert_eq!(OtherStruct.foo(), 456);
}
