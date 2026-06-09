//@ compile-flags: -O

#![feature(f16)]
#![feature(f128)]
#![deny(overflowing_literals)]

fn main() {
    let x2: i8 = --128; //~ ERROR literal out of range for `i8`
    //~| WARN use of a double negation

    let x = -65520.0_f16; //~ ERROR literal out of range for `f16`
    let x =  65520.0_f16; //~ ERROR literal out of range for `f16`
    let x = -3.40282357e+38_f32; //~ ERROR literal out of range for `f32`
    let x =  3.40282357e+38_f32; //~ ERROR literal out of range for `f32`
    let x = -1.7976931348623159e+308_f64; //~ ERROR literal out of range for `f64`
    let x =  1.7976931348623159e+308_f64; //~ ERROR literal out of range for `f64`
    let x = -1.1897314953572317650857593266280075e+4932_f128; //~ ERROR literal out of range for `f128`
    let x =  1.1897314953572317650857593266280075e+4932_f128; //~ ERROR literal out of range for `f128`
}
