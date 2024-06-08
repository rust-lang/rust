// rustfmt-wrap_comments: true
// Test attributes and doc comments are preserved.
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/", test(attr(deny(warnings))))]

//! Doc comment

#![attribute]

//! Crate doc comment

// Comment

// Comment on attribute
#![the(attribute)]

// Another comment

/// Blah blah blah.
/// Blah blah blah.
/// Blah blah blah.
/// Blah blah blah.

/// Blah blah blah.
impl Bar {
    /// Blah blah blooo.
    /// Blah blah blooo.
    /// Blah blah blooo.
    /// Blah blah blooo.
    #[an_attribute]
    #[doc = "an attribute that shouldn't be normalized to a doc comment"]
    fn foo(&mut self) -> isize {
    }

    /// Blah blah bing.
    /// Blah blah bing.
    /// Blah blah bing.


    /// Blah blah bing.
    /// Blah blah bing.
    /// Blah blah bing.
    pub fn f2(self) {
        (foo, bar)
    }

    #[another_attribute]
    fn f3(self) -> Dog {
    }

    /// Blah blah bing.

    #[attrib1]
    /// Blah blah bing.
    #[attrib2]
    // Another comment that needs rewrite because it's tooooooooooooooooooooooooooooooo loooooooooooong.
    /// Blah blah bing.
    fn f4(self) -> Cat {
    }

    // We want spaces around `=`
    #[cfg(feature="nightly")]
    fn f5(self) -> Monkey {}
}

// #984
struct Foo {
    # [ derive ( Clone , PartialEq , Debug , Deserialize , Serialize ) ]
    foo: usize,
}

// #1668

/// Default path (*nix)
#[cfg(all(unix, not(target_os = "macos"), not(target_os = "ios"), not(target_os = "android")))]
fn foo() {
    #[cfg(target_os = "freertos")]
    match port_id {
        'a' | 'A' => GpioPort { port_address: GPIO_A },
        'b' | 'B' => GpioPort { port_address: GPIO_B },
        _ => panic!(),
    }

    #[cfg_attr(not(target_os = "freertos"), allow(unused_variables))]
    let x = 3;
}

// #1777
#[test]
#[should_panic(expected = "(")]
#[should_panic(expected = /* ( */ "(")]
#[should_panic(/* ((((( */expected /* ((((( */= /* ((((( */ "("/* ((((( */)]
#[should_panic(
    /* (((((((( *//*
    (((((((((()(((((((( */
    expected = "("
    // ((((((((
)]
fn foo() {}

// #1799
fn issue_1799() {
    #[allow(unreachable_code)] // https://github.com/rust-lang/rust/issues/43336
    Some( Err(error) ) ;

    #[allow(unreachable_code)]
    // https://github.com/rust-lang/rust/issues/43336
    Some( Err(error) ) ;
}

// Formatting inner attributes
fn inner_attributes() {
    #![ this_is_an_inner_attribute ( foo ) ]

    foo();
}

impl InnerAttributes() {
    #![ this_is_an_inner_attribute ( foo ) ]

    fn foo() {}
}

mod InnerAttributes {
    #![ this_is_an_inner_attribute ( foo ) ]
}

fn attributes_on_statements() {
    // Local
    # [ attr ( on ( local ) ) ]
    let x = 3;

    // Item
    # [ attr ( on ( item ) ) ]
    use foo;

    // Expr
    # [ attr ( on ( expr ) ) ]
    {}

    // Semi
    # [ attr ( on ( semi ) ) ]
    foo();

    // Mac
    # [ attr ( on ( mac ) ) ]
    foo!();
}

// Large derives
#[derive(Add, Sub, Mul, Div, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Serialize, Mul)]


/// Foo bar baz


#[derive(Add, Sub, Mul, Div, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Serialize, Deserialize)]
pub struct HP(pub u8);

// Long `#[doc = "..."]`
struct A { #[doc = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"] b: i32 }

// #2647
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

// #2969
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

// #3313
fn stmt_expr_attributes() {
    let foo ;
    #[must_use]
   foo = false ;
}

// #3509
fn issue3509() {
    match MyEnum {
        MyEnum::Option1 if cfg!(target_os = "windows") =>
            #[cfg(target_os = "windows")]{
                1
            }
    }
    match MyEnum {
        MyEnum::Option1 if cfg!(target_os = "windows") =>
            #[cfg(target_os = "windows")]
                1,
    }
}
