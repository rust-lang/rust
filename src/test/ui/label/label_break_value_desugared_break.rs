// compile-flags: --edition 2018
#![feature(try_blocks)]

// run-pass
fn main() {
    let _: Result<(), ()> = try {
        'foo: {
            Err(())?;
            break 'foo;
        }
    };

    'foo: {
        let _: Result<(), ()> = try {
            Err(())?;
            break 'foo;
        };
    }
}
