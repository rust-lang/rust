// run-pass

#![allow(unreachable_code)]
#![feature(never_type)]

#[allow(unused)]
fn never_returns() {
    loop {
        break loop {};
    }
}

pub fn main() {
    let value = 'outer: loop {
        if 1 == 1 {
            break 13;
        } else {
            let _never: ! = loop {
                break loop {
                    break 'outer panic!();
                }
            };
        }
    };
    assert_eq!(value, 13);

    let x = [1, 3u32, 5];
    let y = [17];
    let z = [];
    let coerced: &[_] = loop {
        match 2 {
            1 => break &x,
            2 => break &y,
            3 => break &z,
            _ => (),
        }
    };
    assert_eq!(coerced, &[17u32]);

    let trait_unified = loop {
        break if true {
            break Default::default()
        } else {
            break [13, 14]
        };
    };
    assert_eq!(trait_unified, [0, 0]);

    let trait_unified_2 = loop {
        if false {
            break [String::from("Hello")]
        } else {
            break Default::default()
        };
    };
    assert_eq!(trait_unified_2, [""]);

    let trait_unified_3 = loop {
        break if false {
            break [String::from("Hello")]
        } else {
            ["Yes".into()]
        };
    };
    assert_eq!(trait_unified_3, ["Yes"]);

    let regular_break: () = loop {
        if true {
            break;
        } else {
            break break Default::default();
        }
    };
    assert_eq!(regular_break, ());

    let regular_break_2: () = loop {
        if true {
            break Default::default();
        } else {
            break;
        }
    };
    assert_eq!(regular_break_2, ());

    let regular_break_3: () = loop {
        break if true {
            Default::default()
        } else {
            break;
        }
    };
    assert_eq!(regular_break_3, ());

    let regular_break_4: () = loop {
        break ();
        break;
    };
    assert_eq!(regular_break_4, ());

    let regular_break_5: () = loop {
        break;
        break ();
    };
    assert_eq!(regular_break_5, ());

    let nested_break_value = 'outer2: loop {
        let _a: u32 = 'inner: loop {
            if true {
                break 'outer2 "hello";
            } else {
                break 'inner 17;
            }
        };
        panic!();
    };
    assert_eq!(nested_break_value, "hello");

    let break_from_while_cond = loop {
        'inner_loop: while break 'inner_loop {
            panic!();
        }
        break 123;
    };
    assert_eq!(break_from_while_cond, 123);

    let break_from_while_to_outer = 'outer_loop: loop {
        while break 'outer_loop 567 {
            panic!("from_inner");
        }
        panic!("from outer");
    };
    assert_eq!(break_from_while_to_outer, 567);

    let rust = true;
    let value = loop {
        break rust;
    };
    assert!(value);
}
