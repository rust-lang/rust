//@ edition: 2018
#![feature(try_blocks)]

//@ check-pass
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
