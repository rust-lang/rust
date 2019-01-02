#[naked]
//~^ the `#[naked]` attribute is an experimental feature
fn naked() {}

#[naked]
//~^ the `#[naked]` attribute is an experimental feature
fn naked_2() -> isize {
    0
}

fn main() {}
