macro_rules! f {
    (v: $x:expr) => ( println!("{}", $x) )
}

fn main () {
    let v = 5;
    f!(v: 3);
}
