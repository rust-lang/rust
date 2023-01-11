fn main() {
    // Should suggest only `std::mem::size_of`
    let _ = size_of::<usize>();
    //~^ ERROR cannot find

    // Should suggest `std::intrinsics::fabsf64`,
    // since there is no non-intrinsic to suggest.
    let _ = fabsf64(1.0);
    //~^ ERROR cannot find
}
