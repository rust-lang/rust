// run-rustfix
#![warn(clippy::cast_abs_to_unsigned)]

fn main() {
    let x: i32 = -42;
    let y: u32 = x.abs() as u32;
    println!("The absolute value of {} is {}", x, y);
}
