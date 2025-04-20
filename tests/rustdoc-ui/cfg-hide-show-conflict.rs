#![feature(doc_cfg)]
#![doc(auto_cfg(hide(target_os = "linux")))]
#![doc(auto_cfg(show(windows, target_os = "linux")))] //~ ERROR
