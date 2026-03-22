// rustfmt-style_edition: 2024

// empty cfg_select!
// Original `{}` delimiters
cfg_select! {
}
std::cfg_select! {
}
core::cfg_select! {
}

// empty with other delimiters
// Original `()` delimiters
cfg_select! (
);
std::cfg_select! (
);
core::cfg_select! (
);

// Original `[]` delimiters
cfg_select! [
];
std::cfg_select! [
];
core::cfg_select! [
];


// Original `{}` delimiters
cfg_select! { /* inline comment */
}
std::cfg_select! { /* inline comment */
}
core::cfg_select! { /* inline comment */
}


// Original `()` delimiters
cfg_select! ( /* inline comment */
);
std::cfg_select! ( /* inline comment */
);
core::cfg_select! ( /* inline comment */
);


// Original `[]` delimiters
cfg_select! [ /* inline comment */
];
std::cfg_select! [ /* inline comment */
];
core::cfg_select! [ /* inline comment */
];


// Original `{}` delimiters
cfg_select! { // opening brace comment

}
std::cfg_select! { // opening brace comment

}
core::cfg_select! { // opening brace comment

}

// Original `()` delimiters
cfg_select! ( // opening brace comment

);
std::cfg_select! ( // opening brace comment

);
core::cfg_select! ( // opening brace comment

);

// Original `[]` delimiters
cfg_select! [ // opening brace comment

];
std::cfg_select! [ // opening brace comment

];
core::cfg_select! [ // opening brace comment

];


// Original `{}` delimiters
cfg_select! {
    // nested inner comment
}
std::cfg_select! {
    // nested inner comment
}
core::cfg_select! {
    // nested inner comment
}

// Original `()` delimiters
cfg_select! (
    // nested inner comment
);
std::cfg_select! (
    // nested inner comment
);
core::cfg_select! (
    // nested inner comment
);

// Original `[]` delimiters
cfg_select! [
    // nested inner comment
];
std::cfg_select! [
    // nested inner comment
];
core::cfg_select! [
    // nested inner comment
];


fn expression_position() {
    // cfg_select arms with block
    println!(cfg_select! {
        unix => { "unix" }
        windows => { "windows" }
        _ => { "not " + "windows" + "or" +       "unix" }
    });

    println!(std::cfg_select! {
        unix => { "unix" }
        windows => { "windows" }
        _ => { "not " + "windows" + "or" +       "unix" }
    });

    println!(core::cfg_select! {
        unix => { "unix" }
        windows => { "windows" }
        _ => { "not " + "windows" + "or" +       "unix" }
    });

    // cfg_select arms with block and trailing commas
    println!(cfg_select! {
        unix => { "unix" },
        windows => { "windows" },
        _ => { "not " + "windows" + "or" +       "unix" },
    });

    println!(std::cfg_select! {
        unix => { "unix" },
        windows => { "windows" },
        _ => { "not " + "windows" + "or" +       "unix" },
    });

    println!(core::cfg_select! {
        unix => { "unix" },
        windows => { "windows" },
        _ => { "not " + "windows" + "or" +       "unix" },
    });

    // cfg_select arms without block
    println!(cfg_select! {
        unix => "unix",
        windows => "windows",
        _ => "not windows or unix",
    });

    println!(std::cfg_select! {
        unix => "unix",
        windows => "windows",
        _ => "not windows or unix",
    });

    println!(core::cfg_select! {
        unix => "unix",
        windows => "windows",
        _ => "not windows or unix",
    });

    // cfg_select arms with and without blocks
    println!(cfg_select! {
        unix => {"unix"}
        windows => {"windows"},
        _ => "not windows or unix",
    });

    println!(std::cfg_select! {
        unix => {"unix"}
        windows => {"windows"},
        _ => "not windows or unix",
    });

    println!(core::cfg_select! {
        unix => {"unix"}
        windows => {"windows"},
        _ => "not windows or unix",
    });
}

// user specified newlines between arms are preserved
core::cfg_select! {
    windows => {}


    unix => {                 }


    _ => {}
}

core::cfg_select! (
    windows => {}


    unix => {                 }


    _ => {}
);

core::cfg_select! [
    windows => {}


    unix => {                 }


    _ => {}
];


// Leading comments are also preserved
core::cfg_select! {
    // windows-Pre Comment

    windows => {}
    // windows-Post comment


    // unix-Pre Comment

    unix => {                 }
    // unix-Post comment


    // wildcard Comment

    _ => {}
    // wildcard-Post comment
}

// trailing comments work
cfg_select! {
    windows => {}
    // windows-Post comment


    unix => {                 }
    // unix-Post comment



    _ => {}
    // wildcard-Post comment
}

core::cfg_select! {
    windows => {}// windows-Post comment


    unix => {                 }// unix-Post comment

    _ => {}// wildcard-Post comment
}

// trailing comments on the last line are a little buggy and always wrap back up
cfg_select! {
    windows => {"windows"}
    unix => {"unix"}
    _ => {"none"}
    // FIXME. Prevent wrapping back up to the next line
}

cfg_select! {
    windows => "windows",
    unix => "unix",
    _ => "none",
    // FIXME. Prevent wrapping back up to the next line
}


// comments within the predicate are fine with style_edition=2024+
cfg_select! {
    any(true, /* comment */
        true, true, // true,
        true, )
        // comment before arrow
        => {}

    not(false // comment
    ) => {

    }

    any(false // comment
    ) => "any"
}

// comments before and after the `=>` get dropped right now
cfg_select! {
    any(true,
        true, true,
        true, )
        // commetn before arrow
        => {}

    not(false
    ) => /* comment before opening brace */ {

    }

    any(false
    ) => // comment before brace
    "any"
}


// A bunch of mixed predicates
cfg_select! {
    // When all predicates are simple uses mixed list formatting
    any(true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true) => {}

    // more "complex" predicates will wrap using vertical formatting
    all(target_arch = "x86_64", true, target_endian = "little", debug_assertions, panic = "unwind", target_env = "gnu") => {}
    all(any(target_arch = "x86_64", true, target_endian = "little"), debug_assertions, panic = "unwind", all(target_env = "gnu", true)) => {}

    // nested "complex" predicates
    all(target_arch = "x86_64", true, target_endian = "little", debug_assertions, panic = "unwind", target_env = "gnu", not(all(target_arch = "x86_64", true, target_endian = "little", debug_assertions, panic = "unwind", target_env = "gnu"))) => {}

    any(true, true, true, true, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true, true, true) => {}
    all(
        any(target_arch = "x86_64", true, target_endian = "little"), debug_assertions,
        panic = "unwind", all(target_env = "gnu", true)
    ) => {}

    all(
        any(target_arch = "x86_64", true, target_endian = "little"),
        debug_assertions,
        panic = "unwind",
        all(target_env = "gnu", true)
    ) => {}

    // This line is under 80 characters, no reason to break.
    any(feature = "acdefg", true, true, true, true, true, true, true, true) => {
        compile_error!("foo")
    }
    // The cfg is under 80 characters, but the line as a whole is over 80 characters.
    any(feature = "acdefgh123", true, true, true, true, true, true, true, true) => {
        compile_error!("foo")
    }
    // The cfg is over 80 characters.
    not(any(feature = "acdefgh1234", true, true, true, true, true, true, true, true)) => {
        compile_error!("foo")
    }
    // make sure that #![feature(cfg_version)] works
    version("1.44.0") => {
    }

    _ => {}
}

// Can't format cfg_select! at all with style_edition <= 2021.
// Things can be formatted with style_edition >= 2024
cfg_select! {
    feature = "debug-with-rustfmt-long-long-long-long-loooooooonnnnnnnnnnnnnnnggggggffffffffffffffff"
    => {
        // abc
        println!(

        );
    }
    feature = "debug-with-rustfmt-long-long-long-long-loooooooonnnnnnnnnnnnnnnggggggffffffffffffffff"
        => {
        // abc
    }
    all(anything("some other long long long long long thing long long long long long long long long long long long", feature = "debug-with-rustfmt-long-long-long-long-loooooooonnnnnnnnnnnnnnnggggggffffffffffffffff")) => {

        let x =    7;
    }
}

std::cfg_select! {
    target_arch = "aarch64" => {
        use std::sync::OnceCell;

        fn foo() {
                return 3;
        }

    }
    _ =>                     {
        compile_error!("mal",   "formed")
    }
    false => {
        compile_error!("also",        "mal",   "formed")
    }
}


mod nested {

// empty cfg_select!
// Original `{}` delimiters
cfg_select! {
}
std::cfg_select! {
}
core::cfg_select! {
}

// empty with other delimiters
// Original `()` delimiters
cfg_select! (
);
std::cfg_select! (
);
core::cfg_select! (
);

// Original `[]` delimiters
cfg_select! [
];
std::cfg_select! [
];
core::cfg_select! [
];


// Original `{}` delimiters
cfg_select! { /* inline comment */
}
std::cfg_select! { /* inline comment */
}
core::cfg_select! { /* inline comment */
}


// Original `()` delimiters
cfg_select! ( /* inline comment */
);
std::cfg_select! ( /* inline comment */
);
core::cfg_select! ( /* inline comment */
);


// Original `[]` delimiters
cfg_select! [ /* inline comment */
];
std::cfg_select! [ /* inline comment */
];
core::cfg_select! [ /* inline comment */
];


// Original `{}` delimiters
cfg_select! { // opening brace comment

}
std::cfg_select! { // opening brace comment

}
core::cfg_select! { // opening brace comment

}

// Original `()` delimiters
cfg_select! ( // opening brace comment

);
std::cfg_select! ( // opening brace comment

);
core::cfg_select! ( // opening brace comment

);

// Original `[]` delimiters
cfg_select! [ // opening brace comment

];
std::cfg_select! [ // opening brace comment

];
core::cfg_select! [ // opening brace comment

];


// Original `{}` delimiters
cfg_select! {
    // nested inner comment
}
std::cfg_select! {
    // nested inner comment
}
core::cfg_select! {
    // nested inner comment
}

// Original `()` delimiters
cfg_select! (
    // nested inner comment
);
std::cfg_select! (
    // nested inner comment
);
core::cfg_select! (
    // nested inner comment
);

// Original `[]` delimiters
cfg_select! [
    // nested inner comment
];
std::cfg_select! [
    // nested inner comment
];
core::cfg_select! [
    // nested inner comment
];


fn expression_position() {
    // cfg_select arms with block
    println!(cfg_select! {
        unix => { "unix" }
        windows => { "windows" }
        _ => { "not " + "windows" + "or" +       "unix" }
    });

    println!(std::cfg_select! {
        unix => { "unix" }
        windows => { "windows" }
        _ => { "not " + "windows" + "or" +       "unix" }
    });

    println!(core::cfg_select! {
        unix => { "unix" }
        windows => { "windows" }
        _ => { "not " + "windows" + "or" +       "unix" }
    });

    // cfg_select arms with block and trailing commas
    println!(cfg_select! {
        unix => { "unix" },
        windows => { "windows" },
        _ => { "not " + "windows" + "or" +       "unix" },
    });

    println!(std::cfg_select! {
        unix => { "unix" },
        windows => { "windows" },
        _ => { "not " + "windows" + "or" +       "unix" },
    });

    println!(core::cfg_select! {
        unix => { "unix" },
        windows => { "windows" },
        _ => { "not " + "windows" + "or" +       "unix" },
    });

    // cfg_select arms without block
    println!(cfg_select! {
        unix => "unix",
        windows => "windows",
        _ => "not windows or unix",
    });

    println!(std::cfg_select! {
        unix => "unix",
        windows => "windows",
        _ => "not windows or unix",
    });

    println!(core::cfg_select! {
        unix => "unix",
        windows => "windows",
        _ => "not windows or unix",
    });

    // cfg_select arms with and without blocks
    println!(cfg_select! {
        unix => {"unix"}
        windows => {"windows"},
        _ => "not windows or unix",
    });

    println!(std::cfg_select! {
        unix => {"unix"}
        windows => {"windows"},
        _ => "not windows or unix",
    });

    println!(core::cfg_select! {
        unix => {"unix"}
        windows => {"windows"},
        _ => "not windows or unix",
    });
}

// user specified newlines between arms are preserved
core::cfg_select! {
    windows => {}


    unix => {                 }


    _ => {}
}

core::cfg_select! (
    windows => {}


    unix => {                 }


    _ => {}
);

core::cfg_select! [
    windows => {}


    unix => {                 }


    _ => {}
];


// Leading comments are also preserved
core::cfg_select! {
    // windows-Pre Comment

    windows => {}
    // windows-Post comment


    // unix-Pre Comment

    unix => {                 }
    // unix-Post comment


    // wildcard Comment

    _ => {}
    // wildcard-Post comment
}

// trailing comments work
cfg_select! {
    windows => {}
    // windows-Post comment


    unix => {                 }
    // unix-Post comment



    _ => {}
    // wildcard-Post comment
}

core::cfg_select! {
    windows => {}// windows-Post comment


    unix => {                 }// unix-Post comment

    _ => {}// wildcard-Post comment
}

// trailing comments on the last line are a little buggy and always wrap back up
cfg_select! {
    windows => {"windows"}
    unix => {"unix"}
    _ => {"none"}
    // FIXME. Prevent wrapping back up to the next line
}

cfg_select! {
    windows => "windows",
    unix => "unix",
    _ => "none",
    // FIXME. Prevent wrapping back up to the next line
}


// comments within the predicate are fine with style_edition=2024+
cfg_select! {
    any(true, /* comment */
        true, true, // true,
        true, )
        // comment before arrow
        => {}

    not(false // comment
    ) => {

    }

    any(false // comment
    ) => "any"
}

// comments before and after the `=>` get dropped right now
cfg_select! {
    any(true,
        true, true,
        true, )
        // commetn before arrow
        => {}

    not(false
    ) => /* comment before opening brace */ {

    }

    any(false
    ) => // comment before brace
    "any"
}


// A bunch of mixed predicates
cfg_select! {
    // When all predicates are simple uses mixed list formatting
    any(true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true) => {}

    // more "complex" predicates will wrap using vertical formatting
    all(target_arch = "x86_64", true, target_endian = "little", debug_assertions, panic = "unwind", target_env = "gnu") => {}
    all(any(target_arch = "x86_64", true, target_endian = "little"), debug_assertions, panic = "unwind", all(target_env = "gnu", true)) => {}

    // nested "complex" predicates
    all(target_arch = "x86_64", true, target_endian = "little", debug_assertions, panic = "unwind", target_env = "gnu", not(all(target_arch = "x86_64", true, target_endian = "little", debug_assertions, panic = "unwind", target_env = "gnu"))) => {}

    any(true, true, true, true, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true, true, true) => {}
    all(
        any(target_arch = "x86_64", true, target_endian = "little"), debug_assertions,
        panic = "unwind", all(target_env = "gnu", true)
    ) => {}

    all(
        any(target_arch = "x86_64", true, target_endian = "little"),
        debug_assertions,
        panic = "unwind",
        all(target_env = "gnu", true)
    ) => {}

    // This line is under 80 characters, no reason to break.
    any(feature = "acdefg", true, true, true, true, true, true, true, true) => {
        compile_error!("foo")
    }
    // The cfg is under 80 characters, but the line as a whole is over 80 characters.
    any(feature = "acdefgh123", true, true, true, true, true, true, true, true) => {
        compile_error!("foo")
    }
    // The cfg is over 80 characters.
    not(any(feature = "acdefgh1234", true, true, true, true, true, true, true, true)) => {
        compile_error!("foo")
    }
    // make sure that #![feature(cfg_version)] works
    version("1.44.0") => {
    }

    _ => {}
}

// Can't format cfg_select! at all with style_edition <= 2021.
// Things can be formatted with style_edition >= 2024
cfg_select! {
    feature = "debug-with-rustfmt-long-long-long-long-loooooooonnnnnnnnnnnnnnnggggggffffffffffffffff"
    => {
        // abc
        println!(

        );
    }
    feature = "debug-with-rustfmt-long-long-long-long-loooooooonnnnnnnnnnnnnnnggggggffffffffffffffff"
        => {
        // abc
    }
    all(anything("some other long long long long long thing long long long long long long long long long long long", feature = "debug-with-rustfmt-long-long-long-long-loooooooonnnnnnnnnnnnnnnggggggffffffffffffffff")) => {

        let x =    7;
    }
}

std::cfg_select! {
    target_arch = "aarch64" => {
        use std::sync::OnceCell;

        fn foo() {
                return 3;
        }

    }
    _ =>                     {
        compile_error!("mal",   "formed")
    }
    false => {
        compile_error!("also",        "mal",   "formed")
    }
}

}


// Some examples I pulled from rust-lang/rust
#[cfg(target_env = "musl")]
cfg_select! {
    all(feature = "llvm-libunwind", feature = "system-llvm-libunwind") => {
        compile_error!("`llvm-libunwind` and `system-llvm-libunwind` cannot be enabled at the same time");
    }
    feature = "llvm-libunwind" => {
        #[link(name = "unwind", kind = "static", modifiers = "-bundle")]
        unsafe extern "C" {}
    }
    feature = "system-llvm-libunwind" => {
        #[link(name = "unwind", kind = "static", modifiers = "-bundle", cfg(target_feature = "crt-static"))]
        #[link(name = "unwind", cfg(not(target_feature = "crt-static")))]
        unsafe extern "C" {}
    }
    _ => {
        #[link(name = "unwind", kind = "static", modifiers = "-bundle", cfg(target_feature = "crt-static"))]
        #[link(name = "gcc_s", cfg(all(not(target_feature = "crt-static"), not(target_arch = "hexagon"))))]
        unsafe extern "C" {}
    }
}

pub const fn midpoint(self, other: f32) -> f32 {
    cfg_select! {
        // Allow faster implementation that have known good 64-bit float
        // implementations. Falling back to the branchy code on targets that don't
        // have 64-bit hardware floats or buggy implementations.
        // https://github.com/rust-lang/rust/pull/121062#issuecomment-2123408114
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(any(target_arch = "riscv32", target_arch = "riscv64"), target_feature = "d"),
            all(target_arch = "loongarch64", target_feature = "d"),
            all(target_arch = "arm", target_feature = "vfp2"),
            target_arch = "wasm32",
            target_arch = "wasm64",
        ) => {
            ((self as f64 + other as f64) / 2.0) as f32
        }
        _ => {
            const HI: f32 = f32::MAX / 2.;

            let (a, b) = (self, other);
            let abs_a = a.abs();
            let abs_b = b.abs();

            if abs_a <= HI && abs_b <= HI {
                // Overflow is impossible
                (a + b) / 2.
            } else {
                (a / 2.) + (b / 2.)
            }
        }
    }
}

mod c_int_definition {
    crate::cfg_select! {
        any(target_arch = "avr", target_arch = "msp430") => {
            pub(super) type c_int = i16;
            pub(super) type c_uint = u16;
        }
        _ => {
            pub(super) type c_int = i32;
            pub(super) type c_uint = u32;
        }
    }
}

cfg_select! {
    any(
        target_family = "unix",
        target_os = "wasi",
        target_os = "teeos",
        target_os = "trusty",
    ) => {
        mod unix;
    }
    target_os = "windows" => {
        mod windows;
    }
    target_os = "hermit" => {
        mod hermit;
    }
    target_os = "motor" => {
        mod motor;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
    }
    target_os = "solid_asp3" => {
        mod solid;
    }
    target_os = "uefi" => {
        mod uefi;
    }
    target_os = "vexos" => {
        mod vexos;
    }
    target_family = "wasm" => {
        mod wasm;
    }
    target_os = "xous" => {
        mod xous;
    }
    target_os = "zkvm" => {
        mod zkvm;
    }
}