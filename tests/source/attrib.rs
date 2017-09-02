// rustfmt-wrap_comments: true
// Test attributes and doc comments are preserved.

//! Doc comment

#![attribute]

//! Crate doc comment

// Comment

// Comment on attribute
#![the(attribute)]

// Another comment

#[invalid attribute]
fn foo() {}

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

// Large derive
#[derive(Add, Sub, Mul, Div, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Serialize, Deserialize)]
pub struct HP(pub u8);
