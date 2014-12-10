rust-clippy
===========

A collection of lints that give helpful tips to newbies.


Lints included in this crate:

 - `clippy_single_match`: Warns when a match statement with a single nontrivial arm (i.e, where the other arm is `_ => {}`) is used, and recommends `if let` instead.
 - `clippy_box_vec`: Warns on usage of `Box<Vec<T>>`
 - `clippy_dlist`: Warns on usage of `DList`

More to come, please [file an issue](https://github.com/Manishearth/rust-clippy/issues) if you have ideas!

Licensed under [MPL](https://www.mozilla.org/MPL/2.0/). If you're having issues with the license, let me know and I'll try to change it to something more permissive.
