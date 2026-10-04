//@ ignore-apple this is supposed to succeed on Apple platforms (though it won't necessarily link)

#[link(name = "foo", kind = "framework")]
extern "C" {}
//~^^ ERROR: link kind `framework` is only supported on Apple targets

fn main() {}
