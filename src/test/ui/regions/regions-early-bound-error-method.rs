// Tests that you can use a fn lifetime parameter as part of
// the value for a type parameter in a bound.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

trait GetRef<'a> {
    fn get(&self) -> &'a isize;
}

struct Box<'a> {
    t: &'a isize
}

impl<'a> GetRef<'a> for Box<'a> {
    fn get(&self) -> &'a isize {
        self.t
    }
}

impl<'a> Box<'a> {
    fn or<'b,G:GetRef<'b>>(&self, g2: G) -> &'a isize {
        g2.get()
        //[base]~^ ERROR E0312
        //[nll]~^^ ERROR lifetime may not live long enough
    }
}

fn main() {
}
