// Checks that you cannot have any item inside `any()` and `none()`.

#![feature(doc_cfg)]

#![doc(auto_cfg(hide(target_os, values(any("linux")))))] //~ ERROR
#![doc(auto_cfg(hide(target_os, values(none("linux")))))] //~ ERROR
