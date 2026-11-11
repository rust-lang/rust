use std::io::Error;

fn main() {
    let _read_num: fn() -> Result<(i32), Error> = || -> Result<(i32), Error> {
        let a = 1;
        Ok(a)
    };
}
