// this test ensures that bad HTML with multiline tags doesn't cause an ICE
// regression test for https://github.com/rust-lang/rust/issues/146890
#[deny(rustdoc::invalid_html_tags)]

/// <TABLE
/// BORDER>
/// <TR
/// >
/// <TH
//~^ ERROR: unclosed HTML tag `TH`
/// >key
/// </TD
//~^ ERROR: unopened HTML tag `TD`
/// >
/// <TH
//~^ ERROR: unclosed HTML tag `TH`
/// >value
/// </TD
//~^ ERROR: unopened HTML tag `TD`
/// >
/// </TR
/// >
/// </TABLE
/// >
pub fn foo() {}
