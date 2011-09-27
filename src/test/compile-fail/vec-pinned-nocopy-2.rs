// error-pattern: mismatched kind

resource r(b: bool) {
}

fn main() {
    let i = [r(true)];
    i += [r(true)];
}