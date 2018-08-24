fn checktrue(rs: bool) -> bool { assert!((rs)); return true; }

pub fn main() { let k = checktrue; evenk(42, k); oddk(45, k); }

fn evenk(n: isize, k: fn(bool) -> bool) -> bool {
    println!("evenk");
    println!("{}", n);
    if n == 0 { return k(true); } else { return oddk(n - 1, k); }
}

fn oddk(n: isize, k: fn(bool) -> bool) -> bool {
    println!("oddk");
    println!("{}", n);
    if n == 0 { return k(false); } else { return evenk(n - 1, k); }
}
