#![warn(clippy::implicit_saturating_sub)]

fn main() {
    let mut end = 10;
    let mut start = 5;
    let mut i: u32 = end - start;

    if i > 0 {
        i -= 1;
    }

    match end {
        10 => {
            if i > 0 {
                i -= 1;
            }
        },
        11 => i += 1,
        _ => i = 0,
    }
}
