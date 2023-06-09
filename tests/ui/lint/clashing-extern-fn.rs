// check-pass
// aux-build:external_extern_fn.rs
#![crate_type = "lib"]
#![warn(clashing_extern_declarations)]

mod redeclared_different_signature {
    mod a {
        extern "C" {
            fn clash(x: u8);
        }
    }
    mod b {
        extern "C" {
            fn clash(x: u64); //~ WARN `clash` redeclared with a different signature
        }
    }
}

mod redeclared_same_signature {
    mod a {
        extern "C" {
            fn no_clash(x: u8);
        }
    }
    mod b {
        extern "C" {
            fn no_clash(x: u8);
        }
    }
}

extern crate external_extern_fn;
mod extern_no_clash {
    // Should not clash with external_extern_fn::extern_fn.
    extern "C" {
        fn extern_fn(x: u8);
    }
}

extern "C" {
    fn some_other_new_name(x: i16);

    #[link_name = "extern_link_name"]
    fn some_new_name(x: i16);

    #[link_name = "link_name_same"]
    fn both_names_different(x: i16);
}

fn link_name_clash() {
    extern "C" {
        fn extern_link_name(x: u32);
        //~^ WARN `extern_link_name` redeclared with a different signature

        #[link_name = "some_other_new_name"]
        //~^ WARN `some_other_extern_link_name` redeclares `some_other_new_name` with a different
        fn some_other_extern_link_name(x: u32);

        #[link_name = "link_name_same"]
        //~^ WARN `other_both_names_different` redeclares `link_name_same` with a different
        fn other_both_names_different(x: u32);
    }
}

mod a {
    extern "C" {
        fn different_mod(x: u8);
    }
}
mod b {
    extern "C" {
        fn different_mod(x: u64); //~ WARN `different_mod` redeclared with a different signature
    }
}

extern "C" {
    fn variadic_decl(x: u8, ...);
}

fn variadic_clash() {
    extern "C" {
        fn variadic_decl(x: u8); //~ WARN `variadic_decl` redeclared with a different signature
    }
}

#[no_mangle]
fn no_mangle_name(x: u8) {}

extern "C" {
    #[link_name = "unique_link_name"]
    fn link_name_specified(x: u8);
}

fn tricky_no_clash() {
    extern "C" {
        // Shouldn't warn, because the declaration above actually declares a different symbol (and
        // Rust's name resolution rules around shadowing will handle this gracefully).
        fn link_name_specified() -> u32;

        // The case of a no_mangle name colliding with an extern decl (see #28179) is related but
        // shouldn't be reported by ClashingExternDeclarations, because this is an example of
        // unmangled name clash causing bad behaviour in functions with a defined body.
        fn no_mangle_name() -> u32;
    }
}

mod banana {
    mod one {
        #[repr(C)]
        struct Banana {
            weight: u32,
            length: u16,
        }
        extern "C" {
            fn weigh_banana(count: *const Banana) -> u64;
        }
    }

    mod two {
        #[repr(C)]
        struct Banana {
            weight: u32,
            length: u16,
        } // note: distinct type
        // This should not trigger the lint because two::Banana is structurally equivalent to
        // one::Banana.
        extern "C" {
            fn weigh_banana(count: *const Banana) -> u64;
        }
    }

    mod three {
        // This _should_ trigger the lint, because repr(packed) should generate a struct that has a
        // different layout.
        #[repr(packed)]
        struct Banana {
            weight: u32,
            length: u16,
        }
        #[allow(improper_ctypes)]
        extern "C" {
            fn weigh_banana(count: *const Banana) -> u64;
            //~^ WARN `weigh_banana` redeclared with a different signature
        }
    }
}

mod sameish_members {
    mod a {
        #[repr(C)]
        struct Point {
            x: i16,
            y: i16,
        }

        extern "C" {
            fn draw_point(p: Point);
        }
    }
    mod b {
        #[repr(C)]
        struct Point {
            coordinates: [i16; 2],
        }

        // It's possible we are overconservative for this case, as accessing the elements of the
        // coordinates array might end up correctly accessing `.x` and `.y`. However, this may not
        // always be the case, for every architecture and situation. This is also a really odd
        // thing to do anyway.
        extern "C" {
            fn draw_point(p: Point);
            //~^ WARN `draw_point` redeclared with a different signature
        }
    }
}

mod same_sized_members_clash {
    mod a {
        #[repr(C)]
        struct Point3 {
            x: f32,
            y: f32,
            z: f32,
        }
        extern "C" {
            fn origin() -> Point3;
        }
    }
    mod b {
        #[repr(C)]
        struct Point3 {
            x: i32,
            y: i32,
            z: i32, // NOTE: Incorrectly redeclared as i32
        }
        extern "C" {
            fn origin() -> Point3; //~ WARN `origin` redeclared with a different signature
        }
    }
}

mod transparent {
    #[repr(transparent)]
    struct T(usize);
    mod a {
        use super::T;
        extern "C" {
            fn transparent() -> T;
            fn transparent_incorrect() -> T;
        }
    }

    mod b {
        extern "C" {
            // Shouldn't warn here, because repr(transparent) guarantees that T's layout is the
            // same as just the usize.
            fn transparent() -> usize;

            // Should warn, because there's a signedness conversion here:
            fn transparent_incorrect() -> isize;
            //~^ WARN `transparent_incorrect` redeclared with a different signature
        }
    }
}

#[allow(improper_ctypes)]
mod zst {
    mod transparent {
        #[repr(transparent)]
        struct TransparentZst(());
        extern "C" {
            fn zst() -> ();
            fn transparent_zst() -> TransparentZst;
        }
    }

    mod not_transparent {
        struct NotTransparentZst(());
        extern "C" {
            // These shouldn't warn since all return types are zero sized
            fn zst() -> NotTransparentZst;
            fn transparent_zst() -> NotTransparentZst;
        }
    }
}

mod missing_return_type {
    mod a {
        extern "C" {
            fn missing_return_type() -> usize;
        }
    }

    mod b {
        extern "C" {
            // This should output a warning because we can't assume that the first declaration is
            // the correct one -- if this one is the correct one, then calling the usize-returning
            // version would allow reads into uninitialised memory.
            fn missing_return_type();
            //~^ WARN `missing_return_type` redeclared with a different signature
        }
    }
}

mod non_zero_and_non_null {
    mod a {
        extern "C" {
            fn non_zero_usize() -> core::num::NonZeroUsize;
            fn non_null_ptr() -> core::ptr::NonNull<usize>;
        }
    }
    mod b {
        extern "C" {
            // If there's a clash in either of these cases you're either gaining an incorrect
            // invariant that the value is non-zero, or you're missing out on that invariant. Both
            // cases are warning for, from both a caller-convenience and optimisation perspective.
            fn non_zero_usize() -> usize;
            //~^ WARN `non_zero_usize` redeclared with a different signature
            fn non_null_ptr() -> *const usize;
            //~^ WARN `non_null_ptr` redeclared with a different signature
        }
    }
}

// See #75739
mod non_zero_transparent {
    mod a1 {
        use std::num::NonZeroUsize;
        extern "C" {
            fn f1() -> NonZeroUsize;
        }
    }

    mod b1 {
        #[repr(transparent)]
        struct X(NonZeroUsize);
        use std::num::NonZeroUsize;
        extern "C" {
            fn f1() -> X;
        }
    }

    mod a2 {
        use std::num::NonZeroUsize;
        extern "C" {
            fn f2() -> NonZeroUsize;
        }
    }

    mod b2 {
        #[repr(transparent)]
        struct X1(NonZeroUsize);

        #[repr(transparent)]
        struct X(X1);

        use std::num::NonZeroUsize;
        extern "C" {
            // Same case as above, but with two layers of newtyping.
            fn f2() -> X;
        }
    }

    mod a3 {
        #[repr(transparent)]
        struct X(core::ptr::NonNull<i32>);

        use std::num::NonZeroUsize;
        extern "C" {
            fn f3() -> X;
        }
    }

    mod b3 {
        extern "C" {
            fn f3() -> core::ptr::NonNull<i32>;
        }
    }

    mod a4 {
        #[repr(transparent)]
        enum E {
            X(std::num::NonZeroUsize),
        }
        extern "C" {
            fn f4() -> E;
        }
    }

    mod b4 {
        extern "C" {
            fn f4() -> std::num::NonZeroUsize;
        }
    }
}

mod null_optimised_enums {
    mod a {
        extern "C" {
            fn option_non_zero_usize() -> usize;
            fn option_non_zero_isize() -> isize;
            fn option_non_null_ptr() -> *const usize;

            fn option_non_zero_usize_incorrect() -> usize;
            fn option_non_null_ptr_incorrect() -> *const usize;
        }
    }
    mod b {
        extern "C" {
            // This should be allowed, because these conversions are guaranteed to be FFI-safe (see
            // #60300)
            fn option_non_zero_usize() -> Option<core::num::NonZeroUsize>;
            fn option_non_zero_isize() -> Option<core::num::NonZeroIsize>;
            fn option_non_null_ptr() -> Option<core::ptr::NonNull<usize>>;

            // However, these should be incorrect (note isize instead of usize)
            fn option_non_zero_usize_incorrect() -> isize;
            //~^ WARN `option_non_zero_usize_incorrect` redeclared with a different signature
            fn option_non_null_ptr_incorrect() -> *const isize;
            //~^ WARN `option_non_null_ptr_incorrect` redeclared with a different signature
        }
    }
}

#[allow(improper_ctypes)]
mod unknown_layout {
    mod a {
        extern "C" {
            pub fn generic(l: Link<u32>);
        }
        pub struct Link<T> {
            pub item: T,
            pub next: *const Link<T>,
        }
    }

    mod b {
        extern "C" {
            pub fn generic(l: Link<u32>);
        }
        pub struct Link<T> {
            pub item: T,
            pub next: *const Link<T>,
        }
    }
}

mod hidden_niche {
    mod a {
        extern "C" {
            fn hidden_niche_transparent() -> usize;
            fn hidden_niche_transparent_no_niche() -> usize;
            fn hidden_niche_unsafe_cell() -> usize;
        }
    }
    mod b {
        use std::cell::UnsafeCell;
        use std::num::NonZeroUsize;

        #[repr(transparent)]
        struct Transparent {
            x: NonZeroUsize,
        }

        #[repr(transparent)]
        struct TransparentNoNiche {
            y: UnsafeCell<NonZeroUsize>,
        }

        extern "C" {
            fn hidden_niche_transparent() -> Option<Transparent>;

            fn hidden_niche_transparent_no_niche() -> Option<TransparentNoNiche>;
            //~^ WARN redeclared with a different signature
            //~| WARN block uses type `Option<TransparentNoNiche>`, which is not FFI-safe

            fn hidden_niche_unsafe_cell() -> Option<UnsafeCell<NonZeroUsize>>;
            //~^ WARN redeclared with a different signature
            //~| WARN block uses type `Option<UnsafeCell<NonZeroUsize>>`, which is not FFI-safe
        }
    }
}
