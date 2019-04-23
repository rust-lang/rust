trait A<T=Self> {}

fn f(a: &A) {}
//~^ ERROR E0393

fn main() {}
