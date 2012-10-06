//error-pattern: by-mutable-reference mode

fn bad(&a: int) {
}

fn called(f: fn(&int)) {
}

fn main() {
called(bad);
}
