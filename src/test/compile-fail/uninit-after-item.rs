// error-pattern:Unsatisfied precondition constraint (for example, init(bar
// xfail-stage0
fn main() {
    auto bar;
    fn baz(int x) { }
    bind baz(bar);
}

