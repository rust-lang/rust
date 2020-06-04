#[inline(always)]
pub fn __incr_cov<T>(_region_loc: &str, result: T) -> T {
    result
}

fn main() {
    let mut countdown = 10;
    __incr_cov("block start",());
    while __incr_cov("while test", countdown > 0) {
        countdown -= 1;
    }

    let mut countdown = 10;
    __incr_cov("after first while loop",());
    while __incr_cov("while test", countdown > 0) {
        countdown -= 1;
        if countdown < 5 {
            __incr_cov("top of if countdown < 5",());
            break;
        }
        countdown -= 2;
    }
}