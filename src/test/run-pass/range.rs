#![allow(unused_comparisons)]
#![allow(dead_code)]
#![allow(unused_mut)]
// Test range syntax.


fn foo() -> isize { 42 }

// Test that range syntax works in return statements
fn return_range_to() -> ::std::ops::RangeTo<i32> { return ..1; }
fn return_full_range() -> ::std::ops::RangeFull { return ..; }

pub fn main() {
    let mut count = 0;
    for i in 0_usize..10 {
        assert!(i >= 0 && i < 10);
        count += i;
    }
    assert_eq!(count, 45);

    let mut count = 0;
    let mut range = 0_usize..10;
    for i in range {
        assert!(i >= 0 && i < 10);
        count += i;
    }
    assert_eq!(count, 45);

    let mut count = 0;
    let mut rf = 3_usize..;
    for i in rf.take(10) {
        assert!(i >= 3 && i < 13);
        count += i;
    }
    assert_eq!(count, 75);

    let _ = 0_usize..4+4-3;
    let _ = 0..foo();

    let _ = { &42..&100 }; // references to literals are OK
    let _ = ..42_usize;

    // Test we can use two different types with a common supertype.
    let x = &42;
    {
        let y = 42;
        let _ = x..&y;
    }
}
