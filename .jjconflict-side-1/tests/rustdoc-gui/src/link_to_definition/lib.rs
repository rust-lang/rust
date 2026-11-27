//@ compile-flags: -Zunstable-options --generate-link-to-definition
pub fn sub_fn() {
    barbar();
}
fn barbar() {
    bar(vec![], vec![], vec![], vec![], Bar { a: "a".into(), b: 0 });
}

pub struct Bar {
    pub a: String,
    pub b: u32,
}

pub fn foo(_b: &Bar) {}

// The goal now is to add
// a lot of lines so
// that the next content
// will be out of the screen
// to allow us to test that
// if the anchor changes to
// something outside of the
// current view, it'll
// scroll to it as expected.

// More filling content.

pub fn bar(
  _a: Vec<String>,
  _b: Vec<String>,
  _c: Vec<String>,
  _d: Vec<String>,
  _e: Bar,
) {
    sub_fn();
}
