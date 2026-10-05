#![doc(auto_cfg)] //~ ERROR
#![doc(auto_cfg(false))]
//~^ ERROR
//~| WARN
#![doc(auto_cfg(true))]
//~^ ERROR
//~| WARN
#![doc(auto_cfg(hide(feature = "solecism")))]
//~^ ERROR
//~| WARN
#![doc(auto_cfg(show(feature = "bla")))]
//~^ ERROR
//~| WARN
#![doc(cfg(feature = "solecism"))]
//~^ ERROR
//~| WARN
