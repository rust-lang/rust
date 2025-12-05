macro_rules! test {
    ($wrong:id) => {};
} //~^ ERROR: invalid fragment specifier `id`

// guard against breaking raw identifier diagnostic
macro_rules! test_raw_identifer {
    ($wrong:r#if) => {};
} //~^ ERROR: invalid fragment specifier `r#if`

fn main() {}
