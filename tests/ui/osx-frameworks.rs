//@ ignore-macos this is supposed to succeed on osx

#[link(name = "foo", kind = "framework")]
extern "C" {}
//~^^ ERROR: link kind `framework` is only supported on Apple targets

fn main() {}
