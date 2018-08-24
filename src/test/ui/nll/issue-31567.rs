// Regression test for #31567: cached results of projections were
// causing region relations not to be enforced at all the places where
// they have to be enforced.

#![feature(nll)]

struct VecWrapper<'a>(&'a mut S);

struct S(Box<u32>);

fn get_dangling<'a>(v: VecWrapper<'a>) -> &'a u32 {
    let s_inner: &'a S = &*v.0; //~ ERROR `*v.0` does not live long enough
    &s_inner.0
}

impl<'a> Drop for VecWrapper<'a> {
    fn drop(&mut self) {
        *self.0 = S(Box::new(0));
    }
}

fn main() {
    let mut s = S(Box::new(11));
    let vw = VecWrapper(&mut s);
    let dangling = get_dangling(vw);
    println!("{}", dangling);
}
