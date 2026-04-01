//@ edition:2024
use submodule::cat; //~ ERROR unresolved import `submodule`
use submodule2::help; //~ ERROR unresolved import `submodule2`
mod success;
fn main() {}
//~? ERROR unresolved import `submodule3`
//~? ERROR unresolved import `submodule4`
