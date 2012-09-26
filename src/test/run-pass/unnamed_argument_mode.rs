fn good(a: &int) {
}

// unnamed argument &int is now parse x: &int

fn called(f: fn(&int)) {
}

fn main() {
called(good);
}
