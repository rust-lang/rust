use NonExistent; //~ ERROR unresolved import `NonExistent`
use non_existent::non_existent; //~ ERROR unresolved import `non_existent`

#[non_existent]
#[derive(NonExistent)]

struct S;

fn main() {}
