// Regression test for #121126. Compiler was panicking when indexing an array
// with an index that is out of bounds and its value is greater than the max
// value allowed for an index.

//@ build-fail

fn main() {
    [0][0xFFFF_FF01];
//~^ ERROR this operation will panic at runtime [unconditional_panic]
}

// NOTE: In order for the test to be valid, the index can take on any value
// between FieldIdx::MAX + 1 (= 0xFFF_FF01) and u32::MAX (= 0xFFF_FFFF)
