// run-rustfix
#![warn(clippy::box_default)]

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
