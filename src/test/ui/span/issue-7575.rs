// Test the mechanism for warning about possible missing `self` declarations.
trait CtxtFn {
    fn f8(self, _: usize) -> usize;
    fn f9(_: usize) -> usize;
}

trait OtherTrait {
    fn f9(_: usize) -> usize;
}

// Note: this trait is not implemented, but we can't really tell
// whether or not an impl would match anyhow without a self
// declaration to match against, so we wind up prisizeing it as a
// candidate. This seems not unreasonable -- perhaps the user meant to
// implement it, after all.
trait UnusedTrait {
    fn f9(_: usize) -> usize;
}

impl CtxtFn for usize {
    fn f8(self, i: usize) -> usize {
        i * 4
    }

    fn f9(i: usize) -> usize {
        i * 4
    }
}

impl OtherTrait for usize {
    fn f9(i: usize) -> usize {
        i * 8
    }
}

struct Myisize(isize);

impl Myisize {
    fn fff(i: isize) -> isize {
        i
    }
}

trait ManyImplTrait {
    fn is_str() -> bool {
        false
    }
}

impl ManyImplTrait for String {
    fn is_str() -> bool {
        true
    }
}

impl ManyImplTrait for usize {}
impl ManyImplTrait for isize {}
impl ManyImplTrait for char {}
impl ManyImplTrait for Myisize {}

fn no_param_bound(u: usize, m: Myisize) -> usize {
    u.f8(42) + u.f9(342) + m.fff(42)
            //~^ ERROR no method named `f9` found for type `usize` in the current scope
            //~| ERROR no method named `fff` found for type `Myisize` in the current scope


}

fn param_bound<T: ManyImplTrait>(t: T) -> bool {
    t.is_str()
    //~^ ERROR no method named `is_str` found for type `T` in the current scope
}

fn main() {
}
