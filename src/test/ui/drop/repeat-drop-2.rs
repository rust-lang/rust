fn main() {
    let foo = String::new();
    let _bar = foo;
    let _baz = [foo; 0]; //~ ERROR use of moved value: `foo` [E0382]
}
