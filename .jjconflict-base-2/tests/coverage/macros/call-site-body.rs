#![feature(coverage_attribute)]
//@ edition: 2024

// Snapshot test demonstrating how the function signature span and body span
// affect coverage instrumentation in the presence of macro expansion.
// This test captures current behaviour, which is not necessarily "correct".

// This macro uses an argument token tree directly as a function body.
#[rustfmt::skip]
macro_rules! with_call_site_body {
    ($body:tt) => {
        fn
        fn_with_call_site_body
            ()
        $body
    }
}

with_call_site_body!(
    // (force line break)
    {
        say("hello");
    }
);

// This macro uses as an argument token tree as code within an explicit body.
#[rustfmt::skip]
macro_rules! with_call_site_inner {
    ($inner:tt) => {
        fn
        fn_with_call_site_inner
            ()
        {
            $inner
        }
    };
}

with_call_site_inner!(
    // (force line break)
    {
        say("hello");
    }
);

#[coverage(off)]
fn main() {
    fn_with_call_site_body();
}

#[coverage(off)]
fn say(message: &str) {
    println!("{message}");
}
