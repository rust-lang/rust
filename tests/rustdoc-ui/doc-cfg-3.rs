// Checks that you cannot have `any()` and values at the same time.

#![deny(invalid_doc_attributes)]
#![feature(doc_cfg)]

#![doc(auto_cfg(hide(target_os, values(any(), "linux"))))] //~ ERROR
#![doc(auto_cfg(hide(target_os, values("linux", any()))))] //~ ERROR
