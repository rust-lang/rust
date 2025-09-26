// Regression test for issue #95879.

use unresolved_crate::module::Name; //~ ERROR cannot find

/// [Name]
pub struct S;
