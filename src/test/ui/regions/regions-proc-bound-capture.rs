// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn borrowed_proc<'a>(x: &'a isize) -> Box<dyn FnMut()->(isize) + 'a> {
    // This is legal, because the region bound on `proc`
    // states that it captures `x`.
    Box::new(move|| { *x })
}

fn static_proc(x: &isize) -> Box<dyn FnMut() -> (isize) + 'static> {
    // This is illegal, because the region bound on `proc` is 'static.
    Box::new(move || { *x })
    //[base]~^ ERROR E0759
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() { }
