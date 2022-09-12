// run-pass
#![allow(dead_code)]
#![allow(unused_assignments)]

// Test control flow to follow label_break_value semantics
fn label_break(a: bool, b: bool) -> u32 {
    let mut v = 0;
    'b: {
        v = 1;
        if a {
            break 'b;
        }
        v = 2;
        if b {
            break 'b;
        }
        v = 3;
    }
    return v;
}

// Test that values can be returned
fn break_value(a: bool, b: bool) -> u32 {
    let result = 'block: {
        if a { break 'block 1; }
        if b { break 'block 2; }
        3
    };
    result
}

// Test nesting of labeled blocks
// here we only check that it compiles
fn label_break_nested() {
    'b: {
        println!("hi");
        if false {
            break 'b;
        }
        'c: {
            if false {
                break 'b;
            }
            break 'c;
        }
        println!("hello");
        if true {
            break 'b;
        }
    }
}

// Tests for mixing labeled blocks with loop constructs
// This function should be the identity function
fn label_break_mixed(v: u32) -> u32 {
    let mut r = 0;
    'b: {
        // Unlabeled break still works
        // (only crossing boundaries is an error)
        loop {
            break;
        }
        if v == 0 {
            break 'b;
        }
        // Labeled breaking an inner loop still works
        'c: loop {
            if r == 1 {
                break 'c;
            }
            r += 1;
        }
        assert_eq!(r, 1);
        if v == 1 {
            break 'b;
        }
        // Labeled breaking an outer loop still works
        'd: loop {
            {
                if v == r {
                    break 'b;
                }
                if r == 5 {
                    break 'd;
                }
                r += 1;
            }
        }
        assert_eq!(r, 5);
        assert!(v > r);
        // Here we test return from inside a labeled block
        return v;
    }
    r
}

fn label_break_match(c: u8, xe: u8, ye: i8) {
    let mut x = 0;
    let y = 'a: {
        match c {
            0 => break 'a 0,
            v if { if v % 2 == 0 { break 'a 1; }; v % 3 == 0 } => { x += 1; },
            v if { 'b: { break 'b v == 5; } } => { x = 41; },
            _ => 'b: {
                break 'b ();
            },
        }
        x += 1;
        -1
    };

    assert_eq!(x, xe);
    assert_eq!(y, ye);
}

#[allow(unused_labels)]
fn label_break_macro() {
    macro_rules! mac1 {
        ($target:lifetime, $val:expr) => {
            break $target $val;
        };
    }
    let x: u8 = 'a: {
        'b: {
            mac1!('b, 1);
        };
        0
    };
    assert_eq!(x, 0);
    let x: u8 = 'a: {
        'b: {
            if true {
                mac1!('a, 1);
            }
        };
        0
    };
    assert_eq!(x, 1);
}

pub fn main() {
    assert_eq!(label_break(true, false), 1);
    assert_eq!(label_break(false, true), 2);
    assert_eq!(label_break(false, false), 3);

    assert_eq!(break_value(true, false), 1);
    assert_eq!(break_value(false, true), 2);
    assert_eq!(break_value(false, false), 3);

    assert_eq!(label_break_mixed(0), 0);
    assert_eq!(label_break_mixed(1), 1);
    assert_eq!(label_break_mixed(2), 2);
    assert_eq!(label_break_mixed(3), 3);
    assert_eq!(label_break_mixed(4), 4);
    assert_eq!(label_break_mixed(5), 5);
    assert_eq!(label_break_mixed(6), 6);

    label_break_match(0, 0, 0);
    label_break_match(1, 1, -1);
    label_break_match(2, 0, 1);
    label_break_match(3, 2, -1);
    label_break_match(5, 42, -1);
    label_break_match(7, 1, -1);

    label_break_macro();
}
