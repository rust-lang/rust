// just make sure this compiles:

fn bar(x: *~int) -> ~int {
    unsafe {
        let y <- *x;
        return y;
    }
}

fn main() {
}