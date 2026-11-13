//@ check-pass
/// Ok: <http://www.unicode.org/reports/tr9/#Reordering_Resolved_Levels>
///
/// Not ok: http://www.unicode.org
/// Not ok: https://www.unicode.org
/// Not ok: http://www.unicode.org/
/// Not ok: http://www.unicode.org/reports/tr9/#Reordering_Resolved_Levels
fn issue_1832() {}

fn main() {}
