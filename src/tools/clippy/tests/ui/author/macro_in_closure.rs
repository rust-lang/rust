//@ check-pass

#![allow(clippy::uninlined_format_args)]

fn main() {
    #[clippy::author]
    let print_text = |x| println!("{}", x);
    print_text("hello");
}
