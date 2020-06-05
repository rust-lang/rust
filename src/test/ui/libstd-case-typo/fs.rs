// checks case typos with libstd::fs structs
fn main(){}

fn test_dirbuild(_x: Dirbuilder){}
//~^ ERROR: cannot find type `Dirbuilder` in this scope
fn test_direntry(_x: Direntry){}
//~^ ERROR: cannot find type `Direntry` in this scope
fn test_filety(_x: Filetype){}
//~^ ERROR: cannot find type `Filetype` in this scope
fn test_metadata(_x: MetaData){}
//~^ ERROR: cannot find type `MetaData` in this scope
fn test_opop(_x: Openoptions){}
//~^ ERROR: cannot find type `Openoptions` in this scope
fn test_perm(_x: permissions){}
//~^ ERROR: cannot find type `permissions` in this scope
fn test_readdir(_x: Readdir){}
//~^ ERROR: cannot find type `Readdir` in this scope
