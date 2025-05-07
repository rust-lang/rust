//@no-rustfix
#![warn(clippy::implicit_saturating_sub)]
#![allow(arithmetic_overflow)]

fn main() {
    let a = 12u32;
    let b = 13u32;

    let result = if a > b { b - a } else { 0 };
    //~^ inverted_saturating_sub

    let result = if b < a { b - a } else { 0 };
    //~^ inverted_saturating_sub

    let result = if a > b { 0 } else { a - b };
    //~^ inverted_saturating_sub

    let result = if a >= b { 0 } else { a - b };
    //~^ inverted_saturating_sub

    let result = if b < a { 0 } else { a - b };
    //~^ inverted_saturating_sub

    let result = if b <= a { 0 } else { a - b };
    //~^ inverted_saturating_sub

    let result = if b * 2 <= a { 0 } else { a - b * 2 };
    //~^ inverted_saturating_sub

    let result = if b <= a * 2 { 0 } else { a * 2 - b };
    //~^ inverted_saturating_sub

    let result = if b + 3 <= a + 2 { 0 } else { (a + 2) - (b + 3) };
    //~^ inverted_saturating_sub

    let af = 12f32;
    let bf = 13f32;
    // Should not lint!
    let result = if bf < af { 0. } else { af - bf };

    // Should not lint!
    let result = if a < b {
        println!("we shouldn't remove this");
        0
    } else {
        a - b
    };
}
