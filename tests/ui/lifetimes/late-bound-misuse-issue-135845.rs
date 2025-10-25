// Regression test for issue #135845
// Ensure we don't ICE when a lifetime parameter from a function
// is incorrectly used in an expression.

struct S<'a, T: ?Sized>(&'a T);

fn b<'a>() -> S<'static, _> {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    S::<'a>(&0)
}

fn static_to_a_to_static_through_ref_in_tuple<'a>(x: &'a u32) -> &'a _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    let (ref y, _z): (&'a u32, u32) = (&22, 44);
    *y
}

fn opt_str2<'a>(maybestr: &'a Option<String>) -> &'a _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    match *maybestr {
        None => "(none)",
        Some(ref s) => {
            let s: &'a str = s;
            s
        }
    }
}

fn main() {}
