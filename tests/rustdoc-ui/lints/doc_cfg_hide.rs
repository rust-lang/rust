#![feature(doc_cfg)]
#![doc(auto_cfg(hide = "test"))] //~ ERROR
#![doc(auto_cfg(hide))] //~ ERROR
#![doc(auto_cfg(hide(not(windows))))] //~ ERROR
