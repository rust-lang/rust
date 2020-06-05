// checks case typos with libstd::iter structs
fn main(){}

fn test_chain(_x: chain<(), ()>){}
//~^ ERROR: cannot find type `chain` in this scope
fn test_cloned(_x: cloned<(), ()>){}
//~^ ERROR: cannot find type `cloned` in this scope
fn test_copied(_x: copied<(), ()>){}
//~^ ERROR: cannot find type `copied` in this scope
fn test_cycle(_x: cycle<(), ()>){}
//~^ ERROR: cannot find type `cycle` in this scope
fn test_empty(_x: empty){}
//~^ ERROR: cannot find type `empty` in this scope
fn test_enumer(_x: enumerate<(), ()>){}
//~^ ERROR: cannot find type `enumerate` in this scope
fn test_filter(_x: filter<(), ()>){}
//~^ ERROR: cannot find type `filter` in this scope
fn test_filtermap(_x: Filtermap<(), ()>){}
//~^ ERROR: cannot find type `Filtermap` in this scope
fn test_flatten(_x: flatten<()>){}
//~^ ERROR: cannot find type `flatten` in this scope
fn test_fromfn(_x: Fromfn<()>){}
//~^ ERROR: cannot find type `Fromfn` in this scope
fn test_fuse(_x: fuse<()>){}
//~^ ERROR: cannot find type `fuse` in this scope
fn test_inspect(_x: inspect<(), ()>){}
//~^ ERROR: cannot find type `inspect` in this scope
fn test_map(_x: map<(), ()>){}
//~^ ERROR: cannot find type `map` in this scope
fn test_once(_x: once<()>){}
//~^ ERROR: cannot find type `once` in this scope
fn test_oncewith(_x: Oncewith<()>){}
//~^ ERROR: cannot find type `Oncewith` in this scope
fn test_peek(_x: peekable<()>){}
//~^ ERROR: cannot find type `peekable` in this scope
fn test_repeat(_x: repeat<()>){}
//~^ ERROR: cannot find type `repeat` in this scope
fn test_repeatw(_x: Repeatwith<()>){}
//~^ ERROR: cannot find type `Repeatwith` in this scope
fn test_rev(_x: rev<()>){}
//~^ ERROR: cannot find type `rev` in this scope
fn test_scan(_x: scan<(), (), ()>){}
//~^ ERROR: cannot find type `scan` in this scope
fn test_skip(_x: skip<()>){}
//~^ ERROR: cannot find type `skip` in this scope
fn test_skipw(_x: Skipwhile<(), ()>){}
//~^ ERROR: cannot find type `Skipwhile` in this scope
fn test_stepby(_x: Stepby<()>){}
//~^ ERROR: cannot find type `Stepby` in this scope
fn test_successors(_x: successors<()>){}
//~^ ERROR: cannot find type `successors` in this scope
fn test_take(_x: take<()>){}
//~^ ERROR: cannot find type `take` in this scope
fn test_takew(_x: Takewhile<(), ()>){}
//~^ ERROR: cannot find type `Takewhile` in this scope
fn test_zip(_x: zip<(), ()>){}
//~^ ERROR: cannot find type `zip` in this scope
