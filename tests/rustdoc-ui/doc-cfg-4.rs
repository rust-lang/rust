// Checks that you cannot have any item inside `any()` and 

#![deny(invalid_doc_attributes)]
#![feature(doc_cfg)]

#![doc(auto_cfg(hide(target_os, values(any("linux")))))] //~ ERROR
#![doc(auto_cfg(hide(target_os, values(none("linux")))))] //~ ERROR
