// Testing that we don't fail abnormally after hitting the errors

import unresolved::*; //~ ERROR unresolved modulename
//~^ ERROR unresolved does not name a module

fn main() {
}