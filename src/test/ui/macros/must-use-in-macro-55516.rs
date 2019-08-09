// build-pass (FIXME(62277): could be check-pass?)
// compile-flags: -Wunused

// make sure write!() can't hide its unused Result

fn main() {
    use std::fmt::Write;
    let mut example = String::new();
    write!(&mut example, "{}", 42); //~WARN must be used
}
