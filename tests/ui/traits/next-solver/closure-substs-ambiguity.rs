//@ compile-flags: -Znext-solver
//@ check-pass

fn main() {
    let mut x: Vec<_> = vec![];
    x.extend(Some(1i32).into_iter().map(|x| x));
}
