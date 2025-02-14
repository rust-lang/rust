use core::fmt::Debug;
use core::mem::size_of;
use std::boxed::ThinBox;

#[test]
fn want_niche_optimization() {
    fn uses_niche<T: ?Sized>() -> bool {
        size_of::<*const ()>() == size_of::<Option<ThinBox<T>>>()
    }

    trait Tr {}
    assert!(uses_niche::<dyn Tr>());
    assert!(uses_niche::<[i32]>());
    assert!(uses_niche::<i32>());
}

#[test]
fn want_thin() {
    fn is_thin<T: ?Sized>() -> bool {
        size_of::<*const ()>() == size_of::<ThinBox<T>>()
    }

    trait Tr {}
    assert!(is_thin::<dyn Tr>());
    assert!(is_thin::<[i32]>());
    assert!(is_thin::<i32>());
}

#[allow(dead_code)]
fn assert_covariance() {
    fn thin_box<'new>(b: ThinBox<[&'static str]>) -> ThinBox<[&'new str]> {
        b
    }
}

#[track_caller]
fn verify_aligned<T>(ptr: *const T) {
    // Use `black_box` to attempt to obscure the fact that we're calling this
    // function on pointers that come from box/references, which the compiler
    // would otherwise realize is impossible (because it would mean we've
    // already executed UB).
    //
    // That is, we'd *like* it to be possible for the asserts in this function
    // to detect brokenness in the ThinBox impl.
    //
    // It would probably be better if we instead had these as debug_asserts
    // inside `ThinBox`, prior to the point where we do the UB. Anyway, in
    // practice these checks are mostly just smoke-detectors for an extremely
    // broken `ThinBox` impl, since it's an extremely subtle piece of code.
    let ptr = core::hint::black_box(ptr);
    assert!(
        ptr.is_aligned() && !ptr.is_null(),
        "misaligned ThinBox data; valid pointers to `{ty}` should be aligned to {align}: {ptr:p}",
        ty = core::any::type_name::<T>(),
        align = core::mem::align_of::<T>(),
    );
}

#[track_caller]
fn check_thin_sized<T: Debug + PartialEq + Clone>(make: impl FnOnce() -> T) {
    let value = make();
    let boxed = ThinBox::new(value.clone());
    let val = &*boxed;
    verify_aligned(val as *const T);
    assert_eq!(val, &value);
}

#[track_caller]
fn check_thin_dyn<T: Debug + PartialEq + Clone>(make: impl FnOnce() -> T) {
    let value = make();
    let wanted_debug = format!("{value:?}");
    let boxed: ThinBox<dyn Debug> = ThinBox::new_unsize(value.clone());
    let val = &*boxed;
    // wide reference -> wide pointer -> thin pointer
    verify_aligned(val as *const dyn Debug as *const T);
    let got_debug = format!("{val:?}");
    assert_eq!(wanted_debug, got_debug);
}

macro_rules! define_test {
    (
        @test_name: $testname:ident;

        $(#[$m:meta])*
        struct $Type:ident($inner:ty);

        $($test_stmts:tt)*
    ) => {
        #[test]
        fn $testname() {
            use core::sync::atomic::{AtomicIsize, Ordering};
            // Define the type, and implement new/clone/drop in such a way that
            // the number of live instances will be counted.
            $(#[$m])*
            #[derive(Debug, PartialEq)]
            struct $Type {
                _priv: $inner,
            }

            impl Clone for $Type {
                fn clone(&self) -> Self {
                    verify_aligned(self);
                    Self::new(self._priv.clone())
                }
            }

            impl Drop for $Type {
                fn drop(&mut self) {
                    verify_aligned(self);
                    Self::modify_live(-1);
                }
            }

            impl $Type {
                fn new(i: $inner) -> Self {
                    Self::modify_live(1);
                    Self { _priv: i }
                }

                fn modify_live(n: isize) -> isize {
                    static COUNTER: AtomicIsize = AtomicIsize::new(0);
                    COUNTER.fetch_add(n, Ordering::Relaxed) + n
                }

                fn live_objects() -> isize {
                    Self::modify_live(0)
                }
            }
            // Run the test statements
            let _: () = { $($test_stmts)* };
            // Check that we didn't leak anything, or call drop too many times.
            assert_eq!(
                $Type::live_objects(), 0,
                "Wrong number of drops of {}, `initializations - drops` should be 0.",
                stringify!($Type),
            );
        }
    };
}

define_test! {
    @test_name: align1zst;
    struct Align1Zst(());

    check_thin_sized(|| Align1Zst::new(()));
    check_thin_dyn(|| Align1Zst::new(()));
}

define_test! {
    @test_name: align1small;
    struct Align1Small(u8);

    check_thin_sized(|| Align1Small::new(50));
    check_thin_dyn(|| Align1Small::new(50));
}

define_test! {
    @test_name: align1_size_not_pow2;
    struct Align64NotPow2Size([u8; 79]);

    check_thin_sized(|| Align64NotPow2Size::new([100; 79]));
    check_thin_dyn(|| Align64NotPow2Size::new([100; 79]));
}

define_test! {
    @test_name: align1big;
    struct Align1Big([u8; 256]);

    check_thin_sized(|| Align1Big::new([5u8; 256]));
    check_thin_dyn(|| Align1Big::new([5u8; 256]));
}

// Note: `#[repr(align(2))]` is worth testing because
// - can have pointers which are misaligned, unlike align(1)
// - is still expected to have an alignment less than the alignment of a vtable.
define_test! {
    @test_name: align2zst;
    #[repr(align(2))]
    struct Align2Zst(());

    check_thin_sized(|| Align2Zst::new(()));
    check_thin_dyn(|| Align2Zst::new(()));
}

define_test! {
    @test_name: align2small;
    #[repr(align(2))]
    struct Align2Small(u8);

    check_thin_sized(|| Align2Small::new(60));
    check_thin_dyn(|| Align2Small::new(60));
}

define_test! {
    @test_name: align2full;
    #[repr(align(2))]
    struct Align2Full([u8; 2]);
    check_thin_sized(|| Align2Full::new([3u8; 2]));
    check_thin_dyn(|| Align2Full::new([3u8; 2]));
}

define_test! {
    @test_name: align2_size_not_pow2;
    #[repr(align(2))]
    struct Align2NotPower2Size([u8; 6]);

    check_thin_sized(|| Align2NotPower2Size::new([3; 6]));
    check_thin_dyn(|| Align2NotPower2Size::new([3; 6]));
}

define_test! {
    @test_name: align2big;
    #[repr(align(2))]
    struct Align2Big([u8; 256]);

    check_thin_sized(|| Align2Big::new([5u8; 256]));
    check_thin_dyn(|| Align2Big::new([5u8; 256]));
}

define_test! {
    @test_name: align64zst;
    #[repr(align(64))]
    struct Align64Zst(());

    check_thin_sized(|| Align64Zst::new(()));
    check_thin_dyn(|| Align64Zst::new(()));
}

define_test! {
    @test_name: align64small;
    #[repr(align(64))]
    struct Align64Small(u8);

    check_thin_sized(|| Align64Small::new(50));
    check_thin_dyn(|| Align64Small::new(50));
}

define_test! {
    @test_name: align64med;
    #[repr(align(64))]
    struct Align64Med([u8; 64]);
    check_thin_sized(|| Align64Med::new([10; 64]));
    check_thin_dyn(|| Align64Med::new([10; 64]));
}

define_test! {
    @test_name: align64_size_not_pow2;
    #[repr(align(64))]
    struct Align64NotPow2Size([u8; 192]);

    check_thin_sized(|| Align64NotPow2Size::new([10; 192]));
    check_thin_dyn(|| Align64NotPow2Size::new([10; 192]));
}

define_test! {
    @test_name: align64big;
    #[repr(align(64))]
    struct Align64Big([u8; 256]);

    check_thin_sized(|| Align64Big::new([10; 256]));
    check_thin_dyn(|| Align64Big::new([10; 256]));
}
