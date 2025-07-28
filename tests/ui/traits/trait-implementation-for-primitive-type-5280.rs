//@ run-pass
#![allow(dead_code)]

type FontTableTag = u32;

trait FontTableTagConversions {
  fn tag_to_string(self);
}

impl FontTableTagConversions for FontTableTag {
  fn tag_to_string(self) {
      let _ = &self;
  }
}

pub fn main() {
    5.tag_to_string();
}

// https://github.com/rust-lang/rust/issues/5280
