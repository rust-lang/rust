// checks case typos with libstd::ops structs
fn main(){}

fn test_range(_x: range<()>){}
//~^ ERROR: cannot find type `range` in this scope
fn test_rangefr(_x: Rangefrom<()>){}
//~^ ERROR: cannot find type `Rangefrom` in this scope
fn test_rangefu(_x: Rangefull<()>){}
//~^ ERROR: cannot find type `Rangefull` in this scope
fn test_rangeinc(_x: Rangeinclusive<()>){}
//~^ ERROR: cannot find type `Rangeinclusive` in this scope
fn test_rangeto(_x: Rangeto<()>){}
//~^ ERROR: cannot find type `Rangeto` in this scope
fn test_rangetoi(_x: RangetoInclusive<()>){}
//~^ ERROR: cannot find type `RangetoInclusive` in this scope
