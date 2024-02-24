//@ check-pass

/// ```{class=language-c}
/// int main(void) { return 0; }
/// ```
//~^^^ WARNING custom classes in code blocks will change behaviour
//~| NOTE found these custom classes: class=language-c
//~| NOTE see issue #79483 <https://github.com/rust-lang/rust/issues/79483>
//~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
//~| HELP add `#![feature(custom_code_classes_in_docs)]` to the crate attributes to enable
pub struct Bar;

/// ```ASN.1
/// int main(void) { return 0; }
/// ```
pub struct Bar2;
