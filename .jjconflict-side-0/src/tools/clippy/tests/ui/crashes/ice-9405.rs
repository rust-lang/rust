//@ check-pass

#![warn(clippy::useless_format)]

fn main() {
    println!(
        "\

            {}",
        "multiple skipped lines"
    );
}
