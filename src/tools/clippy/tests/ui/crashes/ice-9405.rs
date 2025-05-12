//@ check-pass

#![warn(clippy::useless_format)]
#![allow(clippy::print_literal)]

fn main() {
    println!(
        "\

            {}",
        "multiple skipped lines"
    );
}
