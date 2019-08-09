trait A<T=Self> {}

fn f(a: &dyn A) {}
//~^ ERROR E0393

fn main() {}
