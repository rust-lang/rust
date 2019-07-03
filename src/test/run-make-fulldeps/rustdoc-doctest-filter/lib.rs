/// so if we set up our filters to target just this test:
///
/// ```
/// assert!(true);
/// ```
///
/// this test shouldn't run when the makefile is executed:
///
/// ```
/// assert!(false);
/// ```
pub struct SomeStruct;
