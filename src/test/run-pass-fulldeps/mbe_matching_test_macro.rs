// aux-build:procedural_mbe_matching.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(procedural_mbe_matching)]

pub fn main() {
    assert_eq!(matches!(Some(123), None | Some(0)), false);
    assert_eq!(matches!(Some(123), None | Some(123)), true);
    assert_eq!(matches!(true, true), true);
}
