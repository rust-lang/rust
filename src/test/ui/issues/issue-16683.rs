// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

trait T<'a> {
    fn a(&'a self) -> &'a bool;
    fn b(&self) {
        self.a();
        //[base]~^ ERROR cannot infer
        //[nll]~^^ ERROR lifetime may not live long enough
    }
}

fn main() {}
