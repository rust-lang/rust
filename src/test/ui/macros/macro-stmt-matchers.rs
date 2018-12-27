// compile-pass


fn main() {
    macro_rules! m { ($s:stmt;) => { $s } }
    m!(vec![].push(0););
}
