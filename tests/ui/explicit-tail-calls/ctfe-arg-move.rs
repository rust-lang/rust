//@ check-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

pub const fn test(s: String) -> String {
    const fn takes_string(s: String) -> String {
        s
    }

    become takes_string(s);
}

struct Type;

fn main() {}
