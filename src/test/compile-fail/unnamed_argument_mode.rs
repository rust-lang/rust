//error-pattern: mismatched types

fn bad(&a: int) {
}

// unnamed argument &int is now parsed x: &int
// it's not parsed &x: int anymore

fn called(f: fn(&int)) {
}

fn main() {
called(bad);
}
