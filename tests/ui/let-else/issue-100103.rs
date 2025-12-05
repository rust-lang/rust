//@ edition:2021
//@ check-pass

#![feature(try_blocks)]


fn main() {
    let _: Result<i32, i32> = try {
        let Some(x) = Some(0) else {
            Err(1)?
        };

        x
    };
}
