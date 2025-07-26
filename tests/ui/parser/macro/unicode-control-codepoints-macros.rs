// Regression test for #140281
//@ edition: 2021
//@ proc-macro: unicode-control.rs

extern crate unicode_control;
use unicode_control::*;

macro_rules! foo {
    ($x:expr) => {
        $x
    };
}

macro_rules! empty {
    ($x:expr) => {};
}

fn main() {
    let t = vec![
        /// ‮test⁦ RTL in doc in vec
        //~^ ERROR unicode codepoint changing visible direction of text present in doc comment
        1
    ];
    foo!(
        /**
         * ‮test⁦ RTL in doc in macro
         */
        //~^^^ ERROR unicode codepoint changing visible direction of text present in doc comment
        1
    );
    empty!(
        /**
         * ‮test⁦ RTL in doc in macro
         */
        //~^^^ ERROR unicode codepoint changing visible direction of text present in doc comment
        1
    );
    let x = create_rtl_in_string!(); // OK
    forward_stream!(
        /// ‮test⁦ RTL in doc in proc macro
        //~^ ERROR unicode codepoint changing visible direction of text present in doc comment
        mod a {}
    );
    recollect_stream!(
        /// ‮test⁦ RTL in doc in proc macro
        //~^ ERROR unicode codepoint changing visible direction of text present in doc comment
        mod b {}
    );
}
