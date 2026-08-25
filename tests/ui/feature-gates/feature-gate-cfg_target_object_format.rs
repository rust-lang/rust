#[allow(unused)]
#[cfg(target_object_format = "elf")]
//~^ ERROR `cfg(target_object_format)` is experimental
const X: () = ();

fn main() {}
