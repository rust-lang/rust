// pretty-expanded FIXME(#23616)

fn good(_a: &isize) {
}

// Unnamed argument `&isize` is parsed like `x: &isize`.
fn called<F>(_f: F) where F: FnOnce(&isize) {
}

pub fn main() {
    called(good);
}
