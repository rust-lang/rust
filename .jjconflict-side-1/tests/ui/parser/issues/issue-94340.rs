// Make sure that unexpected inner attributes are not labeled as outer ones in diagnostics when
// trying to parse an item and that they are subsequently ignored not triggering confusing extra
// diagnostics like "expected item after attributes" which is not true for `include!` which can
// include empty files.

include!("auxiliary/issue-94340-inc.rs");

fn main() {}

//~? ERROR an inner attribute is not permitted in this context
//~? ERROR an inner attribute is not permitted in this context
