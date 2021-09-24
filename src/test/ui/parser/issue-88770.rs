// Regression test for the ICE described in #88770.

// error-pattern:this file contains an unclosed delimiter
// error-pattern:expected one of
// error-pattern:missing `in` in `for` loop
// error-pattern:expected `;`, found `e`

fn m(){print!("",(c for&g
u
e
e
