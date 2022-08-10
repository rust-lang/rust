// This is a regression test for issue #86208.
// It is also a general test of macro_rules! display.

#![crate_name = "foo"]

// @has 'foo/macro.todo.html'
// @has - '//span[@class="macro"]' 'macro_rules!'
// @has - '//span[@class="ident"]' 'todo'

// @hastext - '{ () =&gt; { ... }; ($('
// @has - '//span[@class="macro-nonterminal"]' '$'
// @has - '//span[@class="macro-nonterminal"]' 'arg'
// @hastext - ':'
// @has - '//span[@class="ident"]' 'tt'
// @hastext - ')+'
// @hastext - ') =&gt; { ... }; }'
pub use std::todo;

mod mod1 {
    // @has 'foo/macro.macro1.html'
    // @hastext - 'macro_rules!'
    // @hastext - 'macro1'
    // @hastext - '{ () =&gt; { ... }; ($('
    // @has - '//span[@class="macro-nonterminal"]' '$'
    // @has - '//span[@class="macro-nonterminal"]' 'arg'
    // @hastext - ':'
    // @hastext - 'expr'
    // @hastext - '),'
    // @hastext - '+'
    // @hastext - ') =&gt; { ... }; }'
    #[macro_export]
    macro_rules! macro1 {
        () => {};
        ($($arg:expr),+) => { stringify!($($arg),+) };
    }
}
