#![feature(doc_cfg)]
#![doc(auto_cfg(hide(target_os, values("linux"))))]
#![doc(auto_cfg(show(windows), show(target_os, values("linux"))))] //~ ERROR
