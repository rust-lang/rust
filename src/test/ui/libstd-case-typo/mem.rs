// checks case typos with libstd::mem structs
fn main(){}

fn test_disc(_x: discriminant<()>){}
//~^ ERROR: cannot find type `discriminant` in this scope
fn test_mandrop(_x: Manuallydrop<()>){}
//~^ ERROR: cannot find type `Manuallydrop` in this scope
