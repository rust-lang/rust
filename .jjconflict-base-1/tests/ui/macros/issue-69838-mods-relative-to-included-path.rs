//@ check-pass

include!("issue-69838-dir/included.rs");

fn main() {
    bar::i_am_in_bar();
}
