// Tests that empty source_maps don't ICE (#23301)

// compile-flags: --cfg ""

// error-pattern: expected identifier, found

pub fn main() {
}
