//@ known-bug: #131294
//@ compile-flags: -Zmir-opt-level=5 -Zvalidate-mir -Zcross-crate-inline-threshold=always

// https://github.com/rust-lang/rust/issues/131294#issuecomment-2395088049 second comment
struct Rows;

impl Iterator for Rows {
    type Item = String;

    fn next() -> Option<String> {
        let args = format_args!("Hello world");

        {
            match args.as_str() {
                Some(t) => t.to_owned(),
                None => String::new(),
            }
        }
            .into()
    }
}

fn main() {
    Rows.next();
}
