// error-pattern:Unsatisfied precondition constraint (for example, init(bar
fn main() {
    let bar;
    fn baz(x: int) { }
    bind baz(bar);
}

