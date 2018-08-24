//

#![warn(overflowing_literals)]
#![warn(const_err)]
#![feature(rustc_attrs)]

#[allow(unused_variables)]
#[rustc_error]
fn main() { //~ ERROR: compilation successful
    let x2: i8 = --128; //~ warn: literal out of range for i8

    let x = -3.40282357e+38_f32; //~ warn: literal out of range for f32
    let x =  3.40282357e+38_f32; //~ warn: literal out of range for f32
    let x = -1.7976931348623159e+308_f64; //~ warn: literal out of range for f64
    let x =  1.7976931348623159e+308_f64; //~ warn: literal out of range for f64
}
