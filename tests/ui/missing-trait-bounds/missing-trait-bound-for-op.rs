// run-rustfix

pub fn foo<T>(s: &[T], t: &[T]) {
    let _ = s == t; //~ ERROR binary operation `==` cannot be applied to type `&[T]`
}

fn main() {}
