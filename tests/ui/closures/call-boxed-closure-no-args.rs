//@ run-pass
// Test that overloaded calls work with zero arity closures


fn main() {
    let functions: [Box<dyn Fn() -> Option<()>>; 1] = [Box::new(|| None)];

    let _: Option<Vec<()>> = functions.iter().map(|f| (*f)()).collect();
}
