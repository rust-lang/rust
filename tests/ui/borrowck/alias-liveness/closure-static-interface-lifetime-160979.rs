//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next

struct Lifetimed<'a> {
    #[allow(dead_code)]
    some_tuple: &'a (),
}
static LIFETIMED: Lifetimed<'static> = Lifetimed { some_tuple: &() };

struct ClosureMaker<'a> {
    #[allow(dead_code)]
    some_string: &'a str,
}

impl<'a> ClosureMaker<'a> {
    // The closure itself is `'static` (it captures nothing), but its return
    // type mentions `'a`.
    fn make_closure_simple(&self) -> impl Fn() -> &'a Lifetimed<'a> + 'static {
        || &LIFETIMED
    }
}

fn leak_it<T: 'static>(val: T) -> &'static T {
    Box::leak(Box::new(val))
}

fn return_value_with_dangling_lifetime() {
    let s = String::from("");
    let cm = ClosureMaker { some_string: &s };
    //~^ ERROR `s` does not live long enough
    let val = cm.make_closure_simple()();
    // requires `&'a Lifetimed<'a>: 'static`
    let _leaked = leak_it(val);
}

fn main() {
    return_value_with_dangling_lifetime();
}
