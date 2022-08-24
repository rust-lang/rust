#![crate_type = "lib"]

fn lbv_macro_test_hygiene_respected() {
    macro_rules! mac2 {
        ($val:expr) => {
            break 'a $val; //~ ERROR undeclared label `'a` [E0426]
        };
    }
    let x: u8 = 'a: {
        'b: {
            if true {
                mac2!(2);
            }
        };
        0
    };
    assert_eq!(x, 2);

    macro_rules! mac3 {
        ($val:expr) => {
            'a: {
                $val
            }
        };
    }
    let x: u8 = mac3!('b: {
        if true {
            break 'a 3; //~ ERROR undeclared label `'a` [E0426]
        }
        0
    });
    assert_eq!(x, 3);
    let x: u8 = mac3!(break 'a 4); //~ ERROR undeclared label `'a` [E0426]
    assert_eq!(x, 4);
}
