// build-pass
// compile-flags: --crate-type=rlib --emit=llvm-ir -Cno-prepopulate-passes

// This is a variant of issue-91050-1.rs -- see there for an explanation.

pub mod before {
    extern "C" {
        pub static GLOBAL1: [u8; 1];
    }

    pub unsafe fn do_something_with_array() -> u8 {
        GLOBAL1[0]
    }
}

pub mod inner {
    extern "C" {
        pub static GLOBAL1: u8;
    }

    pub unsafe fn call() -> u8 {
        GLOBAL1 + 42
    }
}
