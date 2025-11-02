#![doc(auto_cfg)] //~ ERROR
#![doc(auto_cfg(false))] //~ ERROR
#![doc(auto_cfg(true))] //~ ERROR
#![doc(auto_cfg(hide(feature = "solecism")))] //~ ERROR
#![doc(auto_cfg(show(feature = "bla")))] //~ ERROR
#![doc(cfg(feature = "solecism"))] //~ ERROR
