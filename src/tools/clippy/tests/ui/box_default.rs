#![warn(clippy::box_default)]
#![allow(clippy::default_constructed_unit_structs)]

#[derive(Default)]
struct ImplementsDefault;

struct OwnDefault;

impl OwnDefault {
    fn default() -> Self {
        Self
    }
}

macro_rules! outer {
    ($e: expr) => {
        $e
    };
}

fn main() {
    let _string: Box<String> = Box::new(Default::default());
    let _byte = Box::new(u8::default());
    let _vec = Box::new(Vec::<u8>::new());
    let _impl = Box::new(ImplementsDefault::default());
    let _impl2 = Box::new(<ImplementsDefault as Default>::default());
    let _impl3: Box<ImplementsDefault> = Box::new(Default::default());
    let _own = Box::new(OwnDefault::default()); // should not lint
    let _in_macro = outer!(Box::new(String::new()));
    let _string_default = outer!(Box::new(String::from("")));
    let _vec2: Box<Vec<ImplementsDefault>> = Box::new(vec![]);
    let _vec3: Box<Vec<bool>> = Box::new(Vec::from([]));
    let _vec4: Box<_> = Box::new(Vec::from([false; 0]));
    let _more = ret_ty_fn();
    call_ty_fn(Box::new(u8::default()));
    issue_10381();

    // `Box::<Option<_>>::default()` would be valid here, but not `Box::default()` or
    // `Box::<Option<{closure@...}>::default()`
    //
    // Would have a suggestion after https://github.com/rust-lang/rust/blob/fdd030127cc68afec44a8d3f6341525dd34e50ae/compiler/rustc_middle/src/ty/diagnostics.rs#L554-L563
    let mut unnameable = Box::new(Option::default());
    let _ = unnameable.insert(|| {});
}

fn ret_ty_fn() -> Box<bool> {
    Box::new(bool::default())
}

#[allow(clippy::boxed_local)]
fn call_ty_fn(_b: Box<u8>) {
    issue_9621_dyn_trait();
}

use std::io::{Read, Result};

impl Read for ImplementsDefault {
    fn read(&mut self, _: &mut [u8]) -> Result<usize> {
        Ok(0)
    }
}

fn issue_9621_dyn_trait() {
    let _: Box<dyn Read> = Box::new(ImplementsDefault::default());
    issue_10089();
}

fn issue_10089() {
    let _closure = || {
        #[derive(Default)]
        struct WeirdPathed;

        let _ = Box::new(WeirdPathed::default());
    };
}

fn issue_10381() {
    #[derive(Default)]
    pub struct Foo {}
    pub trait Bar {}
    impl Bar for Foo {}

    fn maybe_get_bar(i: u32) -> Option<Box<dyn Bar>> {
        if i % 2 == 0 {
            Some(Box::new(Foo::default()))
        } else {
            None
        }
    }

    assert!(maybe_get_bar(2).is_some());
}

#[allow(unused)]
fn issue_11868() {
    fn foo(_: &mut Vec<usize>) {}

    macro_rules! bar {
        ($baz:expr) => {
            Box::leak(Box::new($baz))
        };
    }

    foo(bar!(vec![]));
    foo(bar!(vec![1]));
}

// Issue #11927: The quickfix for the `Box::new` suggests replacing with `Box::<Inner>::default()`,
// removing the `outer::` segment.
fn issue_11927() {
    mod outer {
        #[derive(Default)]
        pub struct Inner {
            _i: usize,
        }
    }

    fn foo() {
        let _b = Box::new(outer::Inner::default());
        let _b = Box::new(std::collections::HashSet::<i32>::new());
    }
}
