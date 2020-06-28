// check-pass
// aux-build:external_extern_fn.rs
#![crate_type = "lib"]
#![warn(clashing_extern_declarations)]

extern crate external_extern_fn;

extern "C" {
    fn clash(x: u8);
    fn no_clash(x: u8);
}

fn redeclared_different_signature() {
    extern "C" {
        fn clash(x: u64); //~ WARN `clash` redeclared with a different signature
    }

    unsafe {
        clash(123);
        no_clash(123);
    }
}

fn redeclared_same_signature() {
    extern "C" {
        fn no_clash(x: u8);
    }
    unsafe {
        no_clash(123);
    }
}

extern "C" {
    fn extern_fn(x: u64);
}

fn extern_clash() {
    extern "C" {
        fn extern_fn(x: u32); //~ WARN `extern_fn` redeclared with a different signature
    }
    unsafe {
        extern_fn(123);
    }
}

fn extern_no_clash() {
    unsafe {
        external_extern_fn::extern_fn(123);
        crate::extern_fn(123);
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
        extern "C" {
          // This should not trigger the lint because two::Banana is structurally equivalent to
          // one::Banana.
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
            fn draw_point(p: Point); //~ WARN `draw_point` redeclared with a different
        }
    }
}
