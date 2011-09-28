resource r(i: int) {
}

fn main() {
    // Even though this looks like a copy, it's guaranteed not to be
    let a = r(0);
}