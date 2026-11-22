#![doc(auto_cfg)] //~ ERROR
#![doc(auto_cfg(false))]
//~^ ERROR
//~| ERROR
#![doc(auto_cfg(true))]
//~^ ERROR
//~| ERROR
#![doc(auto_cfg(hide(feature = "solecism")))]
//~^ ERROR
//~| ERROR
#![doc(auto_cfg(show(feature = "bla")))]
//~^ ERROR
//~| ERROR
#![doc(cfg(feature = "solecism"))]
//~^ ERROR
//~| WARN
