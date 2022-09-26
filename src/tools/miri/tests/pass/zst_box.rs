fn main() {
    let x = Box::new(());
    let y = Box::new(());
    drop(y);
    let z = Box::new(());
    drop(x);
    drop(z);
}
