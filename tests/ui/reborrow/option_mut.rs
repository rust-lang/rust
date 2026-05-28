fn method(a: Option<&mut ()>) {}

fn main() {
    let a = Some(&mut ());
    let _ = method(a);
    let _ = method(a); //~ERROR use of moved value: `a`
}
