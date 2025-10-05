/*
 * Description: verify 22 and 19 are not classified as exit ppts
 */

fn boop(x: i32) -> i32 {
    let y = 
        if x == 12 {
            22
        } else {
            19
        };
    return y;
}

fn main() {}
