// Testing that we don't fail abnormally after hitting the errors

use unresolved::*; //~ ERROR unresolved name
//~^ ERROR failed to resolve import

fn main() {
}
