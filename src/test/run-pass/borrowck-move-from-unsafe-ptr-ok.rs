// just make sure this compiles:

fn bar(x: *~int) -> ~int {
    unsafe {
        let y = move *x;
        return y;
    }
}

fn main() {
}