use NonExistent; //~ ERROR unresolved import `NonExistent`
use non_existent::non_existent; //~ ERROR unresolved import `non_existent`

#[non_existent] //~ ERROR cannot determine resolution for the attribute macro `non_existent`
#[derive(NonExistent)] //~ ERROR cannot determine resolution for the derive macro `NonExistent`
struct S;

fn main() {}
