//@ check-pass
//
// This tests checks that clashing_extern_declarations handles types that are recursive through a
// pointer or ref argument. See #75512.

#![crate_type = "lib"]

mod raw_ptr_recursion {
    mod a {
        #[repr(C)]
        struct Pointy {
            pointy: *const Pointy,
        }

        extern "C" {
            fn run_pointy(pointy: Pointy);
        }
    }
    mod b {
        #[repr(C)]
        struct Pointy {
            pointy: *const Pointy,
        }

        extern "C" {
            fn run_pointy(pointy: Pointy);
        }
    }
}

mod raw_ptr_recursion_once_removed {
    mod a {
        #[repr(C)]
        struct Pointy1 {
            pointy_two: *const Pointy2,
        }

        #[repr(C)]
        struct Pointy2 {
            pointy_one: *const Pointy1,
        }

        extern "C" {
            fn run_pointy2(pointy: Pointy2);
        }
    }

    mod b {
        #[repr(C)]
        struct Pointy1 {
            pointy_two: *const Pointy2,
        }

        #[repr(C)]
        struct Pointy2 {
            pointy_one: *const Pointy1,
        }

        extern "C" {
            fn run_pointy2(pointy: Pointy2);
        }
    }
}

mod ref_recursion {
    mod a {
        #[repr(C)]
        struct Reffy<'a> {
            reffy: &'a Reffy<'a>,
        }

        extern "C" {
            fn reffy_recursion(reffy: Reffy);
        }
    }
    mod b {
        #[repr(C)]
        struct Reffy<'a> {
            reffy: &'a Reffy<'a>,
        }

        extern "C" {
            fn reffy_recursion(reffy: Reffy);
        }
    }
}

mod ref_recursion_once_removed {
    mod a {
        #[repr(C)]
        struct Reffy1<'a> {
            reffy: &'a Reffy2<'a>,
        }

        #[repr(C)]
        struct Reffy2<'a> {
            reffy: &'a Reffy1<'a>,
        }

        extern "C" {
            #[allow(improper_ctypes)]
            fn reffy_once_removed(reffy: Reffy1);
        }
    }
    mod b {
        #[repr(C)]
        struct Reffy1<'a> {
            reffy: &'a Reffy2<'a>,
        }

        #[repr(C)]
        struct Reffy2<'a> {
            reffy: &'a Reffy1<'a>,
        }

        extern "C" {
            #[allow(improper_ctypes)]
            fn reffy_once_removed(reffy: Reffy1);
        }
    }
}
