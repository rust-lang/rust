//@ check-fail
//
// issue: <https://github.com/rust-lang/rust/issues/120217>

trait Static<'a> {
    fn proof(&self, s: &'a str) -> &'static str;
}

fn bad_cast<'a>(x: *const dyn Static<'static>) -> *const dyn Static<'a> {
    x as _ //~ error: lifetime may not live long enough
}

impl Static<'static> for () {
    fn proof(&self, s: &'static str) -> &'static str {
        s
    }
}

fn extend_lifetime(s: &str) -> &'static str {
    let raw: *const dyn Static<'static> = &() as *const dyn Static<'static>;
    let cast: *const dyn Static<'_> = bad_cast(raw);
    let reference: &dyn Static<'_> = unsafe { &*cast };
    reference.proof(s)
}

fn main() {
    let s = String::from("Hello World");
    let slice = extend_lifetime(&s);
    println!("Now it exists: {slice}");
    drop(s);
    println!("Now it’s gone: {slice}");
}
