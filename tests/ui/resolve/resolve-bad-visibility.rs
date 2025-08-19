//@ edition: 2015
enum E {}
trait Tr {}

pub(in E) struct S; //~ ERROR expected module, found enum `E`
pub(in Tr) struct Z; //~ ERROR expected module, found trait `Tr`
pub(in std::vec) struct F; //~ ERROR visibilities can only be restricted to ancestor modules
pub(in nonexistent) struct G; //~ ERROR failed to resolve
pub(in too_soon) struct H; //~ ERROR failed to resolve

// Visibilities are resolved eagerly without waiting for modules becoming fully populated.
// Visibilities can only use ancestor modules legally which are always available in time,
// so the worst thing that can happen due to eager resolution is a suboptimal error message.
mod too_soon {}

fn main () {}
