fn getbig(i: int) -> int {
    let m = if i >= 0 {
        let j = getbig(i - 1);
        let k = getbig(j - 1);
        k
    } else {
        0
    };
    m
}

fn main() {
    getbig(10000);
}