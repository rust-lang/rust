fn ignore<T>(t: T) {}

fn nested<'x>(x: &'x isize) {
    let y = 3;
    let mut ay = &y; //~ ERROR E0495

    ignore::<Box<dyn for<'z> FnMut(&'z isize)>>(Box::new(|z| {
        ay = x;
        ay = &y;
        ay = z;
    }));

    ignore::< Box<dyn for<'z> FnMut(&'z isize) -> &'z isize>>(Box::new(|z| {
        if false { return x; } //~ ERROR E0312
        if false { return ay; }
        return z;
    }));
}

fn main() {}
