// The diff marker detection was removed for this example, because it relied on
// the lexer having a dependency on the parser, which was horrible.

macro_rules! foo {
<<<<<<< HEAD
    () {
=======
    () { //
>>>>>>> 7a4f13c blah blah blah
    }
} //~ ERROR this file contains an unclosed delimiter
