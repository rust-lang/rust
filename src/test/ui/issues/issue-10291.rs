// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn test<'x>(x: &'x isize) {
    drop::<Box<dyn for<'z> FnMut(&'z isize) -> &'z isize>>(Box::new(|z| {
        x
        //[base]~^ ERROR E0312
        //[nll]~^^ ERROR lifetime may not live long enough
    }));
}

fn main() {}
