// Regression test for issue #124935
// Tests that we do not erroneously emit an error about
// missing main function when the mod starts with a `;`

;
//~^ ERROR expected item, found `;`
fn main() { }
