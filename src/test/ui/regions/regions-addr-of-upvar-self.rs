// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct Dog {
    food: usize,
}

impl Dog {
    pub fn chase_cat(&mut self) {
        let _f = || {
            let p: &'static mut usize = &mut self.food;
            //[base]~^ ERROR cannot infer
            //[nll]~^^ ERROR lifetime may not live long enough
            //[nll]~^^^ ERROR lifetime may not live long enough
            //[nll]~^^^^ ERROR E0597
            *p = 3;
        };
    }
}

fn main() {
}
