// Test that signed and unsigned integer patterns work with `#[loop_match]`.

//@ run-pass

#![allow(incomplete_features)]
#![feature(loop_match)]

fn main() {
    assert_eq!(integer(0), 2);
    assert_eq!(integer(-1), 2);
    assert_eq!(integer(2), 2);

    assert_eq!(boolean(true), false);
    assert_eq!(boolean(false), false);

    assert_eq!(character('a'), 'b');
    assert_eq!(character('b'), 'b');
    assert_eq!(character('c'), 'd');
    assert_eq!(character('d'), 'd');

    assert_eq!(test_f32(1.0), core::f32::consts::PI);
    assert_eq!(test_f32(2.5), core::f32::consts::PI);
    assert_eq!(test_f32(4.0), 4.0);

    assert_eq!(test_f64(1.0), core::f64::consts::PI);
    assert_eq!(test_f64(2.5), core::f64::consts::PI);
    assert_eq!(test_f64(4.0), 4.0);
}

fn integer(mut state: i32) -> i32 {
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                -1 => {
                    #[const_continue]
                    break 'blk 2;
                }
                0 => {
                    #[const_continue]
                    break 'blk -1;
                }
                2 => break 'a,
                _ => unreachable!("weird value {:?}", state),
            }
        }
    }

    state
}

fn boolean(mut state: bool) -> bool {
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                true => {
                    #[const_continue]
                    break 'blk false;
                }
                false => return state,
            }
        }
    }
}

fn character(mut state: char) -> char {
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                'a' => {
                    #[const_continue]
                    break 'blk 'b';
                }
                'b' => return state,
                'c' => {
                    #[const_continue]
                    break 'blk 'd';
                }
                _ => return state,
            }
        }
    }
}

fn test_f32(mut state: f32) -> f32 {
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                1.0 => {
                    #[const_continue]
                    break 'blk 2.5;
                }
                2.0..3.0 => return core::f32::consts::PI,
                _ => return state,
            }
        }
    }
}

fn test_f64(mut state: f64) -> f64 {
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                1.0 => {
                    #[const_continue]
                    break 'blk 2.5;
                }
                2.0..3.0 => return core::f64::consts::PI,
                _ => return state,
            }
        }
    }
}
