// rustfmt-wrap_comments: true
// rustfmt-normalize_doc_attributes: true

//! Example doc attribute comment

// Long `#[doc = "..."]`
struct A {
    /// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    b: i32,
}

/// The `nodes` and `edges` method each return instantiations of `Cow<[T]>` to
/// leave implementers the freedom to create entirely new vectors or to pass
/// back slices into internally owned vectors.
struct B {
    b: i32,
}

/// Level 1 comment
mod tests {
    /// Level 2 comment
    impl A {
        /// Level 3 comment
        fn f() {
            /// Level 4 comment
            fn g() {}
        }
    }
}

struct C {
    /// item doc attrib comment
    // regular item comment
    b: i32,

    // regular item comment
    /// item doc attrib comment
    c: i32,
}

// non-regression test for regular attributes, from #2647
#[cfg(
    feature = "this_line_is_101_characters_long_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)]
pub fn foo() {}

// path attrs
#[clippy::bar]
#[clippy::bar=foo]
#[clippy::bar(a, b, c)]
pub fn foo() {}

mod issue_2620 {
    #[derive(Debug, StructOpt)]
    #[structopt(about = "Display information about the character on FF Logs")]
    pub struct Params {
        #[structopt(help = "The server the character is on")]
        server: String,
        #[structopt(help = "The character's first name")]
        first_name: String,
        #[structopt(help = "The character's last name")]
        last_name: String,
        #[structopt(
            short = "j",
            long = "job",
            help = "The job to look at",
            parse(try_from_str)
        )]
        job: Option<Job>,
    }
}

// non-regression test for regular attributes, from #2969
#[cfg(not(all(
    feature = "std",
    any(
        target_os = "linux",
        target_os = "android",
        target_os = "netbsd",
        target_os = "dragonfly",
        target_os = "haiku",
        target_os = "emscripten",
        target_os = "solaris",
        target_os = "cloudabi",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "bitrig",
        target_os = "redox",
        target_os = "fuchsia",
        windows,
        all(target_arch = "wasm32", feature = "stdweb"),
        all(target_arch = "wasm32", feature = "wasm-bindgen"),
    )
)))]
type Os = NoSource;
