// Regression test for issue #95879.

use unresolved_crate::module::Name; //~ ERROR failed to resolve

/// [Name]
pub struct S;
