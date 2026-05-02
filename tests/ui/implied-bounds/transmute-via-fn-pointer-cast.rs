// Regression test: transmute via fn pointer cast exploiting #25860
// Verifies that lifetime extension through fn pointer variance is rejected.

static UNIT: &'static &'static () = &&();

fn bad<'a, 'b, T: ?Sized>(_: &'a &'b (), v: &'b T) -> &'a T { v }

fn transmute_lifetime<'a, 'b, T: ?Sized>(x: &'a T) -> &'b T {
    let f: fn(_, &'a T) -> &'b T = bad;
    //~^ ERROR lifetime may not live long enough
    f(&&(), x)
}

fn main() {
    let s = String::from("hello");
    let _r: &'static str = transmute_lifetime(&s);
}
