


// -*- rust -*-
mod foo {
    #[legacy_exports];
    fn baz() { }
}

mod bar {
    #[legacy_exports];
    fn baz() { }
}

fn main() { }
