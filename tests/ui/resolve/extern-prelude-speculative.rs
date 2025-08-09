// Non-existent path in `--extern` doesn't result in an error if it's shadowed by `extern crate`.

//@ check-pass
//@ compile-flags: --extern something=/path/to/nowhere

extern crate std as something;

fn main() {
    something::println!();
}
