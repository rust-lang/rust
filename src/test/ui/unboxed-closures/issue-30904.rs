#![feature(fn_traits, unboxed_closures)]

fn test<F: for<'x> FnOnce<(&'x str,)>>(_: F) {}

struct Compose<F,G>(F,G);
impl<T,F,G> FnOnce<(T,)> for Compose<F,G>
where F: FnOnce<(T,)>, G: FnOnce<(F::Output,)> {
    type Output = G::Output;
    extern "rust-call" fn call_once(self, (x,): (T,)) -> G::Output {
        (self.1)((self.0)(x))
    }
}

struct Str<'a>(&'a str);
fn mk_str<'a>(s: &'a str) -> Str<'a> { Str(s) }

fn main() {
    let _: for<'a> fn(&'a str) -> Str<'a> = mk_str;
    // expected concrete lifetime, found bound lifetime parameter 'a
    let _: for<'a> fn(&'a str) -> Str<'a> = Str;
    //~^ ERROR: mismatched types

    test(|_: &str| {});
    test(mk_str);
    // expected concrete lifetime, found bound lifetime parameter 'x
    test(Str); //~ ERROR: type mismatch in function arguments

    test(Compose(|_: &str| {}, |_| {}));
    test(Compose(mk_str, |_| {}));
    // internal compiler error: cannot relate bound region:
    //   ReLateBound(DebruijnIndex { depth: 2 },
    //     BrNamed(DefId { krate: 0, node: DefIndex(6) => test::'x }, 'x(65)))
    //<= ReSkolemized(0,
    //     BrNamed(DefId { krate: 0, node: DefIndex(6) => test::'x }, 'x(65)))
    test(Compose(Str, |_| {}));
}
