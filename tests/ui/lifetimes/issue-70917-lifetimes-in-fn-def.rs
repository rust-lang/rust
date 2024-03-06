//@ check-pass

fn assert_static<T: 'static>(_: T) {}

// NOTE(eddyb) the `'a: 'a` may look a bit strange, but we *really* want
// `'a` to be an *early-bound* parameter, otherwise it doesn't matter anyway.
fn capture_lifetime<'a: 'a>() {}

fn test_lifetime<'a>() {
    assert_static(capture_lifetime::<'a>);
}

fn main() {}
