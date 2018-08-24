// pretty-expanded FIXME #23616

pub fn main() {
    // Make sure that this view item is filtered out because otherwise it would
    // trigger a compilation error
    #[cfg(not_present)] use bar as foo;
}
