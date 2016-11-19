#[deny(warnings)]
fn cfg_return() -> i32 {
    #[cfg(msvc)] return 1;
    #[cfg(not(msvc))] return 2;
}

fn main() {
    cfg_return();
}
