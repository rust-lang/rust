//@ edition: 2015
enum E {}
trait Tr {}

pub(in E) struct S; //~ ERROR expected module, found enum `::E`
pub(in Tr) struct Z; //~ ERROR expected module, found trait `::Tr`
pub(in std::vec) struct F; //~ ERROR visibilities can only be restricted to ancestor modules
pub(in nonexistent) struct G; //~ ERROR cannot find
pub(in too_soon) struct H; //~ ERROR visibilities can only be restricted

// The module exists, but it is not an ancestor module.
mod too_soon {}

fn main () {}
