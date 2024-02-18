//@ run-pass
// check that the derived impls for the comparison traits shortcircuit
// where possible, by having a type that panics when compared as the
// second element, so this passes iff the instances shortcircuit.


use std::cmp::Ordering;

pub struct FailCmp;
impl PartialEq for FailCmp {
    fn eq(&self, _: &FailCmp) -> bool { panic!("eq") }
}

impl PartialOrd for FailCmp {
    fn partial_cmp(&self, _: &FailCmp) -> Option<Ordering> { panic!("partial_cmp") }
}

impl Eq for FailCmp {}

impl Ord for FailCmp {
    fn cmp(&self, _: &FailCmp) -> Ordering { panic!("cmp") }
}

#[derive(PartialEq,PartialOrd,Eq,Ord)]
struct ShortCircuit {
    x: isize,
    y: FailCmp
}

pub fn main() {
    let a = ShortCircuit { x: 1, y: FailCmp };
    let b = ShortCircuit { x: 2, y: FailCmp };

    assert!(a != b);
    assert!(a < b);
    assert_eq!(a.cmp(&b), ::std::cmp::Ordering::Less);
}
