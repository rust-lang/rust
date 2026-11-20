//@check-pass
//@compile-flags: --cfg test

#![warn(clippy::inline_modules)]
#![allow(clippy::non_minimal_cfg)]

fn main() {}

#[cfg(test)]
mod tests {
    mod nested {}
}

#[cfg(all(test))]
mod tests_in_all {
    mod nested {}
}
