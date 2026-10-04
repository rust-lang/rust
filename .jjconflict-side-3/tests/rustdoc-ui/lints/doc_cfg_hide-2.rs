#![feature(doc_cfg)]

#![deny(invalid_doc_attributes)]
#![doc(auto_cfg(hide(not(windows))))] //~ ERROR
