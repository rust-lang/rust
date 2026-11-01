fn main() {
    // Should suggest only `std::mem::transmute`
    let _ = transmute::<usize>();
    //~^ ERROR cannot find

    // Should suggest `std::intrinsics::fabs`,
    // since there is no non-intrinsic to suggest.
    let _ = fabs(1.0);
    //~^ ERROR cannot find
}
