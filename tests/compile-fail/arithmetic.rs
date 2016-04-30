#![feature(plugin)]
#![plugin(clippy)]

#![deny(integer_arithmetic, float_arithmetic)]
#![allow(unused, shadow_reuse, shadow_unrelated, no_effect)]
fn main() {
    let i = 1i32;
    1 + i; //~ERROR integer arithmetic detected
    i * 2; //~ERROR integer arithmetic detected
    1 % //~ERROR integer arithmetic detected
    i / 2; 
    i - 2 + 2 - i; //~ERROR integer arithmetic detected
    -i; //~ERROR integer arithmetic detected
    
    i & 1; // no wrapping
    i | 1; 
    i ^ 1;
    i % 7;
    i >> 1;
    i << 1;
    
    let f = 1.0f32;
    
    f * 2.0; //~ERROR floating-point arithmetic detected
    
    1.0 + f; //~ERROR floating-point arithmetic detected
    f * 2.0; //~ERROR floating-point arithmetic detected
    f / 2.0; //~ERROR floating-point arithmetic detected
    f - 2.0 * 4.2; //~ERROR floating-point arithmetic detected
    -f; //~ERROR floating-point arithmetic detected
}
