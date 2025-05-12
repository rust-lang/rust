//@ check-pass
//@ compile-flags: -Znext-solver

trait Mirror {
    type Assoc;
}

impl<T> Mirror for T {
    type Assoc = T;
}

fn is_static<T: 'static>() {}

fn test<T>() where <T as Mirror>::Assoc: 'static {
    is_static::<T>();
}

fn main() {}
