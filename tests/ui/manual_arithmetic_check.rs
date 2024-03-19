#![warn(clippy::implicit_saturating_sub)]

fn main() {
    let a = 12u32;
    let b = 13u32;
    let c = 8u32;

    let result = if a > b { a - b } else { 0 };
    //~^ ERROR: manual arithmetic check found
    let result = if b < a { a - b } else { 0 };
    //~^ ERROR: manual arithmetic check found

    let result = if a < b { 0 } else { a - b };
    //~^ ERROR: manual arithmetic check found
    let result = if b > a { 0 } else { a - b };
    //~^ ERROR: manual arithmetic check found

    // Should not warn!
    let result = if a > b { a - b } else { a - c };
}
