//@rustc-env: CLIPPY_PETS_PRINT=1
//@rustc-env: CLIPPY_PRINT_MIR=1

#![warn(clippy::borrow_pats)]

fn if_1() {
    if true {
        let _x = 1;
    }
}

fn if_2() {
    if true {
        let _x = 1;
    } else if false {
        let _y = 1;
    }
}

fn loop_1() {
    while !cond_1() {
        while cond_2() {}
    }
}

fn loop_2() {
    let mut idx = 0;
    while idx < 10 {
        idx += 1;
    }
}

fn loop_3() {
    let mut idx = 0;
    loop {
        idx += 1;
        if idx < 10 {
            break;
        }
        let _x = 1;
    }
    let _y = 0;
}

fn block_with_label() -> u32 {
    'label: {
        let _x = 0;
        if !true {
            break 'label;
        }
        let _y = 0;
    }

    12
}

#[allow(clippy::borrow_pats)]
fn loop_4() {
    let mut idx = 0;
    for a in 0..100 {
        for b in 0..100 {
            match (a, b) {
                (1, 2) => break,
                (2, 3) => {
                    let _x = 9;
                },
                (3, _) => {
                    let _y = 8;
                },
                _ => {},
            }
        }
    }
}

fn cond_1() -> bool {
    true
}
fn cond_2() -> bool {
    false
}

fn main() {}
