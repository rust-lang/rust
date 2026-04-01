//@ check-pass
//@ known-bug: #84366

// Should fail. Associated types of 'static types should be `'static`, but
// argument-free closures can be `'static` and return non-`'static` types.

#[allow(dead_code)]
fn foo<'a>() {
    let closure = || -> &'a str { "" };
    assert_static(closure);
}

fn assert_static<T: 'static>(_: T) {}

fn main() {}
