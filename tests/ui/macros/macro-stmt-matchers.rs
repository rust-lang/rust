// build-pass (FIXME(62277): could be check-pass?)


fn main() {
    macro_rules! m { ($s:stmt;) => { $s } }
    m!(vec![].push(0););
}
