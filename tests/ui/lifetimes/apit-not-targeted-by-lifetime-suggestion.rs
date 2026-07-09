// Regression test for #158954.
//
// The "introduce a named lifetime parameter" suggestion must introduce the
// lifetime into the function's own generic parameter list (`fn f<'a>(...)`)
// and must never insert it into an unrelated argument-position `impl Trait`
// parameter (which would produce invalid code like `_fun: 'a, impl Clone` or
// `_fun: &'a, impl Clone`).

fn push_apit(vec: &mut Vec<&str>, s: &str, _fun: impl Clone) {
    vec.push(s);
    //~^ ERROR lifetime may not live long enough
}

fn push_ref_apit(vec: &mut Vec<&str>, s: &str, _fun: &impl Clone) {
    vec.push(s);
    //~^ ERROR lifetime may not live long enough
}

fn push_real_generic<T>(vec: &mut Vec<&str>, s: &str, _fun: T) {
    vec.push(s);
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
