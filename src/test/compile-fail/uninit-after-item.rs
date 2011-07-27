// error-pattern:Unsatisfied precondition constraint (for example, init(bar
// xfail-stage0
fn main() {
    let bar;
    fn baz(x: int) { }
    bind baz(bar);
}

