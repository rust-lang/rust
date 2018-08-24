use core::cmp::Ordering::{Less, Greater, Equal};

#[test]
fn test_int_totalord() {
    assert_eq!(5.cmp(&10), Less);
    assert_eq!(10.cmp(&5), Greater);
    assert_eq!(5.cmp(&5), Equal);
    assert_eq!((-5).cmp(&12), Less);
    assert_eq!(12.cmp(&-5), Greater);
}

#[test]
fn test_mut_int_totalord() {
    assert_eq!((&mut 5).cmp(&&mut 10), Less);
    assert_eq!((&mut 10).cmp(&&mut 5), Greater);
    assert_eq!((&mut 5).cmp(&&mut 5), Equal);
    assert_eq!((&mut -5).cmp(&&mut 12), Less);
    assert_eq!((&mut 12).cmp(&&mut -5), Greater);
}

#[test]
fn test_ord_max_min() {
    assert_eq!(1.max(2), 2);
    assert_eq!(2.max(1), 2);
    assert_eq!(1.min(2), 1);
    assert_eq!(2.min(1), 1);
    assert_eq!(1.max(1), 1);
    assert_eq!(1.min(1), 1);
}

#[test]
fn test_ordering_reverse() {
    assert_eq!(Less.reverse(), Greater);
    assert_eq!(Equal.reverse(), Equal);
    assert_eq!(Greater.reverse(), Less);
}

#[test]
fn test_ordering_order() {
    assert!(Less < Equal);
    assert_eq!(Greater.cmp(&Less), Greater);
}

#[test]
fn test_ordering_then() {
    assert_eq!(Equal.then(Less), Less);
    assert_eq!(Equal.then(Equal), Equal);
    assert_eq!(Equal.then(Greater), Greater);
    assert_eq!(Less.then(Less), Less);
    assert_eq!(Less.then(Equal), Less);
    assert_eq!(Less.then(Greater), Less);
    assert_eq!(Greater.then(Less), Greater);
    assert_eq!(Greater.then(Equal), Greater);
    assert_eq!(Greater.then(Greater), Greater);
}

#[test]
fn test_ordering_then_with() {
    assert_eq!(Equal.then_with(|| Less), Less);
    assert_eq!(Equal.then_with(|| Equal), Equal);
    assert_eq!(Equal.then_with(|| Greater), Greater);
    assert_eq!(Less.then_with(|| Less), Less);
    assert_eq!(Less.then_with(|| Equal), Less);
    assert_eq!(Less.then_with(|| Greater), Less);
    assert_eq!(Greater.then_with(|| Less), Greater);
    assert_eq!(Greater.then_with(|| Equal), Greater);
    assert_eq!(Greater.then_with(|| Greater), Greater);
}

#[test]
fn test_user_defined_eq() {
    // Our type.
    struct SketchyNum {
        num : isize
    }

    // Our implementation of `PartialEq` to support `==` and `!=`.
    impl PartialEq for SketchyNum {
        // Our custom eq allows numbers which are near each other to be equal! :D
        fn eq(&self, other: &SketchyNum) -> bool {
            (self.num - other.num).abs() < 5
        }
    }

    // Now these binary operators will work when applied!
    assert!(SketchyNum {num: 37} == SketchyNum {num: 34});
    assert!(SketchyNum {num: 25} != SketchyNum {num: 57});
}
