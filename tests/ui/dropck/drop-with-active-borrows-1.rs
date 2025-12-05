fn main() {
    let a = "".to_string();
    let b: Vec<&str> = a.lines().collect();
    drop(a);    //~ ERROR cannot move out of `a` because it is borrowed
    for s in &b {
        println!("{}", *s);
    }
}
