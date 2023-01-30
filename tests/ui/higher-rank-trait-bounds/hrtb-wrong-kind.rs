fn a() where for<T> T: Copy {}
//~^ ERROR only lifetime parameters can be used in this context

fn b() where for<const C: usize> [(); C]: Copy {}
//~^ ERROR only lifetime parameters can be used in this context

fn main() {}
