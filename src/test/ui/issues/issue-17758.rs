// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

// Test that regionck suggestions in a provided method of a trait
// don't ICE

trait Foo<'a> {
    fn foo(&'a self);
    fn bar(&self) {
        self.foo();
        //[base]~^ ERROR cannot infer
        //[nll]~^^ ERROR lifetime may not live long enough
    }
}

fn main() {}
