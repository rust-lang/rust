// This test ensures that when a macro call expands to multiple items,
// all of them are present in the macro expansion.
// Regression test for <https://github.com/rust-lang/rust/issues/157508>.

//@ compile-flags: -Zunstable-options --generate-macro-expansion

#![crate_name = "foo"]

macro_rules! print_skip {
    ($($t: ty),* $(,)?) => {$
        (impl $t { fn f() {} })*
    };
}

pub struct A;
pub struct B;

//@ has 'src/foo/multiple-items-in-one-macro-call.rs.html'
// Both `impl A` and `impl B` should be here.
//@ matches - '//*[@class="expansion"]/*[@class="expanded"]' \
//    '^impl A \{\s+fn f\(\) \{\}\n\}\nimpl B \{\s+fn f\(\) \{\}\n\}$'
print_skip!(A, B);
