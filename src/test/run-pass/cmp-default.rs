use std::cmp::Ordering;

// Test default methods in PartialOrd and PartialEq
//
#[derive(Debug)]
struct Fool(bool);

impl PartialEq for Fool {
    fn eq(&self, other: &Fool) -> bool {
        let Fool(this) = *self;
        let Fool(other) = *other;
        this != other
    }
}

struct Int(isize);

impl PartialEq for Int {
    fn eq(&self, other: &Int) -> bool {
        let Int(this) = *self;
        let Int(other) = *other;
        this == other
    }
}

impl PartialOrd for Int {
    fn partial_cmp(&self, other: &Int) -> Option<Ordering> {
        let Int(this) = *self;
        let Int(other) = *other;
        this.partial_cmp(&other)
    }
}

struct RevInt(isize);

impl PartialEq for RevInt {
    fn eq(&self, other: &RevInt) -> bool {
        let RevInt(this) = *self;
        let RevInt(other) = *other;
        this == other
    }
}

impl PartialOrd for RevInt {
    fn partial_cmp(&self, other: &RevInt) -> Option<Ordering> {
        let RevInt(this) = *self;
        let RevInt(other) = *other;
        other.partial_cmp(&this)
    }
}

pub fn main() {
    assert!(Int(2) >  Int(1));
    assert!(Int(2) >= Int(1));
    assert!(Int(1) >= Int(1));
    assert!(Int(1) <  Int(2));
    assert!(Int(1) <= Int(2));
    assert!(Int(1) <= Int(1));

    assert!(RevInt(2) <  RevInt(1));
    assert!(RevInt(2) <= RevInt(1));
    assert!(RevInt(1) <= RevInt(1));
    assert!(RevInt(1) >  RevInt(2));
    assert!(RevInt(1) >= RevInt(2));
    assert!(RevInt(1) >= RevInt(1));

    assert_eq!(Fool(true), Fool(false));
    assert!(Fool(true)  != Fool(true));
    assert!(Fool(false) != Fool(false));
    assert_eq!(Fool(false), Fool(true));
}
