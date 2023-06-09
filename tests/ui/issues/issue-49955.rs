// run-pass

const ALL_THE_NUMS: [u32; 1] = [
    1
];

#[inline(never)]
fn array(i: usize) -> &'static u32 {
    return &ALL_THE_NUMS[i];
}

#[inline(never)]
fn tuple_field() -> &'static u32 {
    &(42,).0
}

fn main() {
    assert_eq!(tuple_field().to_string(), "42");
    assert_eq!(array(0).to_string(), "1");
}
