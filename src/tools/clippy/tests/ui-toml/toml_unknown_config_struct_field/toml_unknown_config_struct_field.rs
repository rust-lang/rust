#[rustfmt::skip]
//@error-in-other-file: error reading Clippy's configuration file: data did not match any variant of untagged enum DisallowedPathEnum
fn main() {
    panic!();
}
