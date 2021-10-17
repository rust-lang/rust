// run-rustfix

pub fn foo<T>(s: &[T], t: &[T]) {
    let _ = s == t; //~ ERROR can't compare
}

fn main() {}
