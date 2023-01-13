// run-rustfix

#![warn(clippy::cast_abs_to_unsigned)]
#![allow(clippy::uninlined_format_args, unused)]

fn main() {
    let x: i32 = -42;
    let y: u32 = x.abs() as u32;
    println!("The absolute value of {} is {}", x, y);

    let a: i32 = -3;
    let _: usize = a.abs() as usize;
    let _: usize = a.abs() as _;
    let _ = a.abs() as usize;

    let a: i64 = -3;
    let _ = a.abs() as usize;
    let _ = a.abs() as u8;
    let _ = a.abs() as u16;
    let _ = a.abs() as u32;
    let _ = a.abs() as u64;
    let _ = a.abs() as u128;

    let a: isize = -3;
    let _ = a.abs() as usize;
    let _ = a.abs() as u8;
    let _ = a.abs() as u16;
    let _ = a.abs() as u32;
    let _ = a.abs() as u64;
    let _ = a.abs() as u128;

    let _ = (x as i64 - y as i64).abs() as u32;
}

#[clippy::msrv = "1.50"]
fn msrv_1_50() {
    let x: i32 = 10;
    assert_eq!(10u32, x.abs() as u32);
}

#[clippy::msrv = "1.51"]
fn msrv_1_51() {
    let x: i32 = 10;
    assert_eq!(10u32, x.abs() as u32);
}
