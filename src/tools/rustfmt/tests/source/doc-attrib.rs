// rustfmt-wrap_comments: true
// rustfmt-normalize_doc_attributes: true

// Only doc = "" attributes should be normalized
#![doc = " Example doc attribute comment"]
#![doc = "          Example doc attribute comment with 10 leading spaces"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/", test(attr(deny(warnings))))]


// Long `#[doc = "..."]`
struct A { #[doc = " xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"] b: i32 }


#[doc = " The `nodes` and `edges` method each return instantiations of `Cow<[T]>` to leave implementers the freedom to create entirely new vectors or to pass back slices into internally owned vectors."]
struct B { b: i32 }


#[doc = " Level 1 comment"]
mod tests {
    #[doc = " Level 2 comment"]
    impl A {
        #[doc = " Level 3 comment"]
        fn f() {
            #[doc = " Level 4 comment"]
            fn g() {
            }
        }
    }
}

struct C {
    #[doc = " item doc attrib comment"]
    // regular item comment
    b: i32,

    // regular item comment
    #[doc = " item doc attrib comment"]
    c: i32,
}

// non-regression test for regular attributes, from #2647
#[cfg(feature = "this_line_is_101_characters_long_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")]
pub fn foo() {}

// path attrs
#[clippy::bar]
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
  job: Option<Job>
}
}

// non-regression test for regular attributes, from #2969
#[cfg(not(all(feature="std",
              any(target_os = "linux", target_os = "android",
                  target_os = "netbsd",
                  target_os = "dragonfly",
                  target_os = "haiku",
                  target_os = "emscripten",
                  target_os = "solaris",
                  target_os = "cloudabi",
                  target_os = "macos", target_os = "ios",
                  target_os = "freebsd",
                  target_os = "openbsd",
                  target_os = "redox",
                  target_os = "fuchsia",
                  windows,
                  all(target_arch = "wasm32", feature = "stdweb"),
                  all(target_arch = "wasm32", feature = "wasm-bindgen"),
              ))))]
type Os = NoSource;

// use cases from bindgen needing precise control over leading spaces
#[doc = " <div rustbindgen accessor></div>"]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct ContradictAccessors {
    #[doc = "<foo>no leading spaces here</foo>"]
    pub mBothAccessors: ::std::os::raw::c_int,
    #[doc = " <div rustbindgen accessor=\"false\"></div>"]
    pub mNoAccessors: ::std::os::raw::c_int,
    #[doc = " <div rustbindgen accessor=\"unsafe\"></div>"]
    pub mUnsafeAccessors: ::std::os::raw::c_int,
    #[doc = " <div rustbindgen accessor=\"immutable\"></div>"]
    pub mImmutableAccessor: ::std::os::raw::c_int,
}

#[doc = " \\brief          MPI structure"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mbedtls_mpi {
    #[doc = "<  integer sign"]
    pub s: ::std::os::raw::c_int,
    #[doc = "<  total # of limbs"]
    pub n: ::std::os::raw::c_ulong,
    #[doc = "<  pointer to limbs"]
    pub p: *mut mbedtls_mpi_uint,
}
