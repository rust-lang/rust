// xfail-test
// Testing that we don't fail abnormally after hitting the errors

use unresolved::*; //~ ERROR unresolved modulename
//~^ ERROR unresolved does not name a module

fn main() {
}
