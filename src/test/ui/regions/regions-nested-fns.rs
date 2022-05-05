// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn ignore<T>(t: T) {}

fn nested<'x>(x: &'x isize) {
    let y = 3;
    let mut ay = &y;
    //[base]~^ ERROR E0495
    //[nll]~^^ ERROR `y` does not live long enough [E0597]

    ignore::<Box<dyn for<'z> FnMut(&'z isize)>>(Box::new(|z| {
        ay = x;
        ay = &y;
        //[nll]~^ ERROR `y` does not live long enough
        ay = z;
        //[nll]~^ ERROR borrowed data escapes outside of closure [E0521]
    }));

    ignore::< Box<dyn for<'z> FnMut(&'z isize) -> &'z isize>>(Box::new(|z| {
        if false { return x; }
        //[base]~^ ERROR E0312
        //[nll]~^^ ERROR lifetime may not live long enough
        if false { return ay; }
        return z;
    }));
}

fn main() {}
