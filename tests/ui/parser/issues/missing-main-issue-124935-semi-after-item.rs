// Regression test for issue #124935
// Tests that we still emit an error after an item.

fn main() { }
;
//~^ ERROR expected item, found `;`
