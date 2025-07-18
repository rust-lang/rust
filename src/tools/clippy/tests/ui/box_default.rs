#![warn(clippy::box_default)]
#![allow(clippy::boxed_local, clippy::default_constructed_unit_structs)]

#[derive(Default)]
struct ImplementsDefault;

struct OwnDefault;

impl OwnDefault {
    fn default() -> Self {
        Self
    }
}

macro_rules! default {
    () => {
        Default::default()
    };
}

macro_rules! string_new {
    () => {
        String::new()
    };
}

macro_rules! box_new {
    ($e:expr) => {
        Box::new($e)
    };
}

fn main() {
    let string1: Box<String> = Box::new(Default::default());
    //~^ box_default
    let string2: Box<String> = Box::new(String::new());
    //~^ box_default
    let impl1: Box<ImplementsDefault> = Box::new(Default::default());
    //~^ box_default
    let vec: Box<Vec<u8>> = Box::new(Vec::new());
    //~^ box_default
    let byte: Box<u8> = Box::new(u8::default());
    //~^ box_default
    let vec2: Box<Vec<ImplementsDefault>> = Box::new(vec![]);
    //~^ box_default
    let vec3: Box<Vec<bool>> = Box::new(Vec::from([]));
    //~^ box_default

    let plain_default = Box::new(Default::default());
    //~^ box_default
    let _: Box<String> = plain_default;

    let _: Box<String> = Box::new(default!());
    let _: Box<String> = Box::new(string_new!());
    let _: Box<String> = box_new!(Default::default());
    let _: Box<String> = box_new!(String::new());
    let _: Box<String> = box_new!(default!());
    let _: Box<String> = box_new!(string_new!());

    let own: Box<OwnDefault> = Box::new(OwnDefault::default()); // should not lint

    // Do not suggest where a turbofish would be required
    let impl2 = Box::new(ImplementsDefault::default());
    let impl3 = Box::new(<ImplementsDefault as Default>::default());
    let vec4: Box<_> = Box::new(Vec::from([false; 0]));
    let more = ret_ty_fn();
    call_ty_fn(Box::new(u8::default()));
    //~^ box_default
    issue_10381();

    // `Box::<Option<_>>::default()` would be valid here, but not `Box::default()` or
    // `Box::<Option<{closure@...}>::default()`
    //
    // Would have a suggestion after https://github.com/rust-lang/rust/blob/fdd030127cc68afec44a8d3f6341525dd34e50ae/compiler/rustc_middle/src/ty/diagnostics.rs#L554-L563
    let mut unnameable = Box::new(Option::default());
    let _ = unnameable.insert(|| {});

    let _ = Box::into_raw(Box::new(String::default()));
}

fn ret_ty_fn() -> Box<bool> {
    Box::new(bool::default()) // Could lint, currently doesn't
}

fn call_ty_fn(_b: Box<u8>) {
    issue_9621_dyn_trait();
}

struct X<T>(T);

impl<T: Default> X<T> {
    fn x(_: Box<T>) {}

    fn same_generic_param() {
        Self::x(Box::new(T::default()));
        //~^ box_default
    }
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
        if i.is_multiple_of(2) {
            Some(Box::new(Foo::default()))
        } else {
            None
        }
    }

    assert!(maybe_get_bar(2).is_some());
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
