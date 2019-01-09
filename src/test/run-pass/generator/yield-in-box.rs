// run-pass

// Test that box-statements with yields in them work.

#![feature(generators, box_syntax)]

fn main() {
    let x = 0i32;
    || {
        let y = 2u32;
        {
            let _t = box (&x, yield 0, &y);
        }
        match box (&x, yield 0, &y) {
            _t => {}
        }
    };
}
