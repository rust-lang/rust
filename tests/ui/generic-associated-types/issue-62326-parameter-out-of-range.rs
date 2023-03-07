// check-pass

trait Iterator {
    type Item<'a>: 'a;
}

impl Iterator for () {
    type Item<'a> = &'a ();
}

fn main() {}
