// This is a regression test for issue #86208.
// It is also a general test of macro_rules! display.

#![crate_name = "foo"]

// @has 'foo/macro.todo.html'
// @has - '//span[@class="macro"]' 'macro_rules!'
// @has - '//span[@class="ident"]' 'todo'
// Note: the only op is the `+`
// @count - '//pre[@class="rust macro"]//span[@class="op"]' 1

// @has - '{ () =&gt; { ... }; ($('
// @has - '//span[@class="macro-nonterminal"]' '$'
// @has - '//span[@class="macro-nonterminal"]' 'arg'
// @has - ':'
// @has - '//span[@class="ident"]' 'tt'
// @has - '),'
// @has - '//span[@class="op"]' '+'
// @has - ') =&gt; { ... }; }'
pub use std::todo;

mod mod1 {
    // @has 'foo/macro.macro1.html'
    // @has - 'macro_rules!'
    // @has - 'macro1'
    // @has - '{ () =&gt; { ... }; ($('
    // @has - '//span[@class="macro-nonterminal"]' '$'
    // @has - '//span[@class="macro-nonterminal"]' 'arg'
    // @has - ':'
    // @has - 'expr'
    // @has - '),'
    // @has - '+'
    // @has - ') =&gt; { ... }; }'
    #[macro_export]
    macro_rules! macro1 {
        () => {};
        ($($arg:expr),+) => { stringify!($($arg),+) };
    }
}
