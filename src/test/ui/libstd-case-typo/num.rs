// checks case typos with libstd::num structs
fn main(){}

fn test_nzi8(_x: NonZeroi8){}
//~^ ERROR: cannot find type `NonZeroi8` in this scope
fn test_nzi16(_x: NonZeroi16){}
//~^ ERROR: cannot find type `NonZeroi16` in this scope
fn test_nzi32(_x: NonZeroi32){}
//~^ ERROR: cannot find type `NonZeroi32` in this scope
fn test_nzi64(_x: NonZeroi64){}
//~^ ERROR: cannot find type `NonZeroi64` in this scope
fn test_nzi128(_x: NonZeroi128){}
//~^ ERROR: cannot find type `NonZeroi128` in this scope
fn test_nzu8(_x: NonZerou8){}
//~^ ERROR: cannot find type `NonZerou8` in this scope
fn test_nzu16(_x: NonZerou16){}
//~^ ERROR: cannot find type `NonZerou16` in this scope
fn test_nzu32(_x: NonZerou32){}
//~^ ERROR: cannot find type `NonZerou32` in this scope
fn test_nzu64(_x: NonZerou64){}
//~^ ERROR: cannot find type `NonZerou64` in this scope
fn test_nzu128(_x: NonZerou128){}
//~^ ERROR: cannot find type `NonZerou128` in this scope
fn test_nzus(_x: NonzeroUsize){}
//~^ ERROR: cannot find type `NonzeroUsize` in this scope
fn test_wrap(_x: wrapping){}
//~^ ERROR: cannot find type `wrapping` in this scope
