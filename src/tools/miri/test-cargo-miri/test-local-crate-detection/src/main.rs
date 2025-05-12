fn main() {
    // Make sure we detect all crates from this workspace as "local".
    // The env var is set during the "build" so we can use `env!` to access it directly.
    println!("{}", env!("MIRI_LOCAL_CRATES"));
}
