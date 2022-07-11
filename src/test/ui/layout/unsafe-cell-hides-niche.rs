// For rust-lang/rust#68303: the contents of `UnsafeCell<T>` cannot
// participate in the niche-optimization for enum discriminants. This
// test checks that an `Option<UnsafeCell<NonZeroU32>>` has the same
// size in memory as an `Option<UnsafeCell<u32>>` (namely, 8 bytes).

// check-pass

#![feature(repr_simd)]

use std::cell::{UnsafeCell, RefCell, Cell};
use std::mem::size_of;
use std::num::NonZeroU32 as N32;
use std::sync::{Mutex, RwLock};

struct Wrapper<T>(T);

#[repr(transparent)]
struct Transparent<T>(T);

struct NoNiche<T>(UnsafeCell<T>);

struct Size<const S: usize>;

// Overwriting the runtime assertion and making it a compile-time assertion
macro_rules! assert_size {
    ($a:ty, $b:expr) => {{
        const _: Size::<{$b}> = Size::<{size_of::<$a>()}>;
    }};
}

const PTR_SIZE: usize = std::mem::size_of::<*const ()>();

fn main() {
    assert_size!(Option<Wrapper<u32>>,     8);
    assert_size!(Option<Wrapper<N32>>,     4); // (✓ niche opt)
    assert_size!(Option<Transparent<u32>>, 8);
    assert_size!(Option<Transparent<N32>>, 4); // (✓ niche opt)
    assert_size!(Option<NoNiche<u32>>,     8);
    assert_size!(Option<NoNiche<N32>>,     8); // (✗ niche opt)

    assert_size!(Option<UnsafeCell<u32>>,  8);
    assert_size!(Option<UnsafeCell<N32>>,  8); // (✗ niche opt)

    assert_size!(       UnsafeCell<&()> , PTR_SIZE);
    assert_size!(Option<UnsafeCell<&()>>, PTR_SIZE * 2); // (✗ niche opt)
    assert_size!(             Cell<&()> , PTR_SIZE);
    assert_size!(Option<      Cell<&()>>, PTR_SIZE * 2); // (✗ niche opt)
    assert_size!(          RefCell<&()> , PTR_SIZE * 2);
    assert_size!(Option<   RefCell<&()>>, PTR_SIZE * 3); // (✗ niche opt)
    assert_size!(
        RwLock<&()>,
        if cfg!(target_pointer_width = "32") { 16 } else { 24 }
    );
    assert_size!(
        Option<RwLock<&()>>,
        if cfg!(target_pointer_width = "32") { 20 } else { 32 }
    ); // (✗ niche opt)
    assert_size!(
        Mutex<&()> ,
        if cfg!(target_pointer_width = "32") { 12 } else { 16 }
    );
    assert_size!(
        Option<Mutex<&()>>,
        if cfg!(target_pointer_width = "32") { 16 } else { 24 }
    ); // (✗ niche opt)

    assert_size!(       UnsafeCell<&[i32]> , PTR_SIZE * 2);
    assert_size!(Option<UnsafeCell<&[i32]>>, PTR_SIZE * 3); // (✗ niche opt)
    assert_size!(       UnsafeCell<(&(), &())> , PTR_SIZE * 2);
    assert_size!(Option<UnsafeCell<(&(), &())>>, PTR_SIZE * 3); // (✗ niche opt)

    trait Trait {}
    assert_size!(       UnsafeCell<&dyn Trait> , PTR_SIZE * 2);
    assert_size!(Option<UnsafeCell<&dyn Trait>>, PTR_SIZE * 3); // (✗ niche opt)

    #[repr(simd)]
    pub struct Vec4<T>([T; 4]);

    assert_size!(       UnsafeCell<Vec4<N32>> , 16);
    assert_size!(Option<UnsafeCell<Vec4<N32>>>, 32); // (✗ niche opt)
}
