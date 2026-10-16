cfg_if::cfg_if! {
    if #[cfg(unix)] {
        fn foo {}
    }
}

fn main() {
    foo!(1   +   2);
    foo!(1 + 2);
}
