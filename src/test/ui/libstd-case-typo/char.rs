// checks case typos with libstd::char structs
fn main(){}

fn test_du16(_x: DecodeUTF16<()>){}
//~^ ERROR: cannot find type `DecodeUTF16` in this scope

fn test_edflt(_x: Escapedefault){}
//~^ ERROR: cannot find type `Escapedefault` in this scope

fn test_euni(_x: Escapeunicode){}
//~^ ERROR: cannot find type `Escapeunicode` in this scope

fn test_tolow(_x: Tolowercase){}
//~^ ERROR: cannot find type `Tolowercase` in this scope

fn test_toupper(_x: Touppercase){}
//~^ ERROR: cannot find type `Touppercase` in this scope
