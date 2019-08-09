// run-pass
// Make sure this compiles without getting a linker error because of missing
// drop-glue because the collector missed adding drop-glue for the closure:

fn create_fn() -> Box<dyn Fn()> {
    let text = String::new();

    Box::new(move || { let _ = &text; })
}

fn main() {
    let _ = create_fn();
}
