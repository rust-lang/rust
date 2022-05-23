// Tests that you can use a fn lifetime parameter as part of
// the value for a type parameter in a bound.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

trait GetRef<'a, T> {
    fn get(&self) -> &'a T;
}

struct Box<'a, T:'a> {
    t: &'a T
}

impl<'a,T:Clone> GetRef<'a,T> for Box<'a,T> {
    fn get(&self) -> &'a T {
        self.t
    }
}

fn get<'a,'b,G:GetRef<'a, isize>>(g1: G, b: &'b isize) -> &'b isize {
    g1.get()
    //[base]~^ ERROR E0312
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {
}
