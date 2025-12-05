//@ build-fail

fn main() {
    recurse(std::iter::empty::<()>())
}

fn recurse(nums: impl Iterator) {
    if true { return }

    recurse(nums.skip(42).peekable())
    //~^ ERROR: reached the recursion limit while instantiating
}
