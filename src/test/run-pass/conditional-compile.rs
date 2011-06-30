// xfail-stage0
// xfail-stage1
// xfail-stage2

// Since the bogus configuration isn't defined main will just be
// parsed, but nothing further will be done with it
#[cfg(bogus)]
fn main() { fail }

fn main() {}