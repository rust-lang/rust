//@run-rustfix
#![warn(clippy::map_identity)]
#![allow(clippy::needless_return)]

fn main() {
    let x: [u16; 3] = [1, 2, 3];
    // should lint
    let _: Vec<_> = x.iter().map(not_identity).map(|x| return x).collect();
    let _: Vec<_> = x.iter().map(std::convert::identity).map(|y| y).collect();
    let _: Option<u8> = Some(3).map(|x| x);
    let _: Result<i8, f32> = Ok(-3).map(|x| {
        return x;
    });
    // should not lint
    let _: Vec<_> = x.iter().map(|x| 2 * x).collect();
    let _: Vec<_> = x.iter().map(not_identity).map(|x| return x - 4).collect();
    let _: Option<u8> = None.map(|x: u8| x - 1);
    let _: Result<i8, f32> = Err(2.3).map(|x: i8| {
        return x + 3;
    });
    let _: Result<u32, u32> = Ok(1).map_err(|a| a);
    let _: Result<u32, u32> = Ok(1).map_err(|a: u32| a * 42);
}

fn not_identity(x: &u16) -> u16 {
    *x
}
