// xfail-stage1
// xfail-stage2
// xfail-stage3
// error-pattern:quux
fn main() {
    let x: int = do { fail "quux" } while (true);
}
