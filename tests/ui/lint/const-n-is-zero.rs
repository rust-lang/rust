// Test that core functions annotated with `#[rustc_panics_when_zero]` lint when `N` is zero

//@ build-fail

#![feature(iter_map_windows)]
#![feature(iter_array_chunks)]

const ZERO: usize = 0;
const ONE: usize = 1;

fn main() {
    let s = [1, 2, 3, 4];

    let _ = s.array_windows::<0>();
    //~^ ERROR this operation will panic at runtime
    //~| NOTE `#[deny(unconditional_panic)]` on by default
    //~| NOTE const parameter `N` is zero

    let _ = s.as_chunks::<{ 0 }>();
    //~^ ERROR this operation will panic at runtime
    //~| NOTE const parameter `N` is zero
    //
    let _ = s.as_rchunks::<{ 1 - 1 }>();
    //~^ ERROR this operation will panic at runtime
    //~| NOTE const parameter `N` is zero

    let mut m = [1, 2, 3, 4];

    let _ = m.as_chunks_mut::<ZERO>();
    //~^ ERROR this operation will panic at runtime
    //~| NOTE const parameter `N` is zero

    let _ = m.as_rchunks_mut::<{ if ZERO == 0 { 0 } else { 1 } }>();
    //~^ ERROR this operation will panic at runtime
    //~| NOTE const parameter `N` is zero

    let _ = s.array_windows().any(|[]| true);
    //~^ ERROR this operation will panic at runtime
    //~| NOTE const parameter `N` is zero

    let _ = s.iter().map_windows(|[]| true);
    //~^ ERROR this operation will panic at runtime
    //~| NOTE const parameter `N` is zero

    let _ = s.iter().array_chunks().any(|[]| true);
    //~^ ERROR this operation will panic at runtime
    //~| NOTE const parameter `N` is zero

    // Shouldn't lint
    let _ = s.array_windows::<2>();
    let _ = s.as_chunks::<1>();
    let _ = m.as_chunks_mut::<ONE>();
    let _ = m.as_rchunks::<{ 1 + 1 }>();
    let _ = m.as_rchunks_mut::<{ if ZERO == 1 { 0 } else { 5 } }>();
}
