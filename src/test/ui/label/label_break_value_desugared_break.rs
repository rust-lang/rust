// compile-flags: --edition 2018
#![feature(label_break_value, try_blocks)]

// run-pass
fn main() {
    let _: Result<(), ()> = try {
        'foo: {
            Err(())?;
            break 'foo;
        }
    };
}
