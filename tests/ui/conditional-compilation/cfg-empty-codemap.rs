// Tests that empty source_maps don't ICE (#23301)

// compile-flags: --error-format=human --cfg ""

// error-pattern: invalid `--cfg` argument: `""` (expected `key` or `key="value"`)

pub fn main() {
}
