// Regression test for ICE #126359

// Tests that there is no ICE when the generic const
// specifying the size of an array is of a non-usize type

struct OppOrder<const N: u8 = 3, T = u32> {
    arr: [T; N], // Array size is u8 instead of usize
    //~^ ERROR mismatched types
}

fn main() {
    let _ = OppOrder::<3, u32> { arr: [0, 0, 0] };
    //~^ ERROR mismatched types
}
