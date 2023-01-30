// gate-test-const_try

const fn t() -> Option<()> {
    Some(())?;
    //~^ error: `?` is not allowed in a `const fn`
    None
}

fn main() {}
