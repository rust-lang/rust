//@ known-bug: #118590

fn main() {
    recurse(std::iter::empty::<()>())
}

fn recurse(nums: impl Iterator) {
    if true { return }

    recurse(nums.skip(42).peekable())
}
