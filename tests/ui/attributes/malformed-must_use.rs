#[must_use()] //~ ERROR valid forms for the attribute are `#[must_use = "reason"]` and `#[must_use]`
struct Test;

fn main() {}
