// gate-test-const_try

const fn t() -> Option<()> {
    Some(())?;
    //~^ error: `?` is not allowed in a `const fn`
    //~| ERROR: cannot convert
    //~| ERROR: cannot determine
    None
}

fn main() {}
