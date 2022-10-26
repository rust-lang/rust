// check-pass
struct NeedsCopy<T: Copy>(T);

// Skips WF because of an escaping bound region.
struct HasWfHigherRanked
where
    (for<'a> fn(NeedsCopy<&'a mut u32>)):,
{}

// Skips WF because of a placeholder region.
struct HasWfPlaceholder
where
    for<'a> NeedsCopy<&'a mut u32>:,
{}

fn main() {
    let _: HasWfHigherRanked;
    let _: HasWfPlaceholder;
}
