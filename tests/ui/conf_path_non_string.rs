#![feature(attr_literals)]
#![feature(plugin)]
#![plugin(clippy(conf_file=42))]
//~^ ERROR `conf_file` value must be a string

fn main() {}
