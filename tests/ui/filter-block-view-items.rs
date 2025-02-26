//@ run-pass

pub fn main() {
    // Make sure that this view item is filtered out because otherwise it would
    // trigger a compilation error
    #[cfg(FALSE)] use bar as foo;
}
