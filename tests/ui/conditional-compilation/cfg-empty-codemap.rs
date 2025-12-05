// Tests that empty source_maps don't ICE (#23301)

//@ compile-flags: --cfg ""

pub fn main() {
}

//~? ERROR invalid `--cfg` argument: `""` (expected `key` or `key="value"`)
