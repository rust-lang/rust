rust-clippy
===========

A collection of lints that give helpful tips to newbies.


Lints included in this crate:

 - `clippy_single_match`: Warns when a match statement with a single nontrivial arm (i.e, where the other arm is `_ => {}`) is used, and recommends `if let` instead.
 - `clippy_box_vec`: Warns on usage of `Box<Vec<T>>`

More to come, please [file an issue](https://github.com/Manishearth/rust-clippy/issues) if you have ideas!