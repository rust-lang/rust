#![crate_type = "rlib"]
#![feature(link_arg_attribute)]

#[link(name = "-weak_framework", kind = "link-arg", modifiers = "+verbatim")]
#[link(name = "CoreFoundation", kind = "link-arg", modifiers = "+verbatim")]
extern "C" {}
