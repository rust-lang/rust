// build-pass (FIXME(62277): could be check-pass?)

fn main() {
    let _ = &[("", ""); 3];
}

const FOO: &[(&str, &str)] = &[("", ""); 3];
const BAR: &[(&str, &str); 5] = &[("", ""); 5];
const BAA: &[[&str; 12]; 11] = &[[""; 12]; 11];
