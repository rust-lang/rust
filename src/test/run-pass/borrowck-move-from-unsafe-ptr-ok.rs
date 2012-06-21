// just make sure this compiles:

fn bar(x: *~int) -> ~int {
    unsafe {
        let y <- *x;
        ret y;
    }
}

fn main() {
}