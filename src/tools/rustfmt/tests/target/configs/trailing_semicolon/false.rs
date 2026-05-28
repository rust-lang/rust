// rustfmt-trailing_semicolon: false

#![feature(loop_break_value)]

fn main() {
    'a: loop {
        break 'a
    }

    let mut done = false;
    'b: while !done {
        done = true;
        continue 'b
    }

    let x = loop {
        break 5
    };

    let x = 'c: loop {
        break 'c 5
    };
}

fn foo() -> usize {
    return 0
}
