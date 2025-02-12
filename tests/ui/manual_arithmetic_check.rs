#![warn(clippy::implicit_saturating_sub, clippy::inverted_saturating_sub)]
#![allow(clippy::if_same_then_else)]

fn main() {
    let a = 12u32;
    let b = 13u32;
    let c = 8u32;

    let result = if a > b { a - b } else { 0 };
    //~^ implicit_saturating_sub

    let result = if b < a { a - b } else { 0 };
    //~^ implicit_saturating_sub

    let result = if a < b { 0 } else { a - b };
    //~^ implicit_saturating_sub

    let result = if b > a { 0 } else { a - b };
    //~^ implicit_saturating_sub

    // Should not warn!
    let result = if a > b { a - b } else { a - c };

    // Just to check it won't break clippy.
    let result = if b > a { 0 } else { 0 };
}
