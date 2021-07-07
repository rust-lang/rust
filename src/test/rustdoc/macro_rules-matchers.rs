// This is a regression test for issue #86208.
// It is also a general test of macro_rules! display.

#![crate_name = "foo"]

// @has 'foo/macro.todo.html'
// @has - '//span[@class="macro"]' 'macro_rules!'
// @has - '//span[@class="ident"]' 'todo'
// Note: count = 2 * ('=' + '>') + '+' = 2 * (1 + 1) + 1 = 5
// @count - '//pre[@class="rust macro"]//span[@class="op"]' 5

// @has - '{ ()'
// @has - '//span[@class="op"]' '='
// @has - '//span[@class="op"]' '>'
// @has - '{ ... };'

// @has - '($('
// @has - '//span[@class="macro-nonterminal"]' '$'
// @has - '//span[@class="macro-nonterminal"]' 'arg'
// @has - ':'
// @has - '//span[@class="ident"]' 'tt'
// @has - '),'
// @has - '//span[@class="op"]' '+'
// @has - ')'
pub use std::todo;

mod mod1 {
    // @has 'foo/macro.macro1.html'
    // @has - 'macro_rules!'
    // @has - 'macro1'
    // @has - '{ ()'
    // @has - '($('
    // @has - '//span[@class="macro-nonterminal"]' '$'
    // @has - '//span[@class="macro-nonterminal"]' 'arg'
    // @has - ':'
    // @has - 'expr'
    // @has - '),'
    // @has - '+'
    // @has - ')'
    #[macro_export]
    macro_rules! macro1 {
        () => {};
        ($($arg:expr),+) => { stringify!($($arg),+) };
    }
}
