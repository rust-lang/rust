// Regression test for ICE #126359

// Tests that there is no ICE when the generic const
// specifying the size of an array is of a non-usize type

struct OppOrder<const N: u8 = 3, T = u32> {
    arr: [T; N],
    //~^ ERROR the constant `N` is not of type `usize`
}

fn main() {
    let _ = OppOrder::<3, u32> { arr: [0, 0, 0] };
    //~^ ERROR the constant `3` is not of type `usize`
}
