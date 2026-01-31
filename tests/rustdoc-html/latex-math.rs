#![crate_name = "foo"]

//@ hasraw foo/bar/index.html '<math>'
/// test math $\frac{ d }{ d x } \tan x = \frac{ 1 }{ \cos^2 x }$
pub mod bar {}

//@ hasraw foo/baz/index.html '<math display="block">'
/// test math $$\frac{ d }{ d x } \tan x = \frac{ 1 }{ \cos^2 x }$$
pub mod baz {}

//@ !hasraw foo/frob/index.html '<math>'
//@ !hasraw foo/frob/index.html '<math display="block">'
/// not math $\frac{ d }
pub mod frob {}
