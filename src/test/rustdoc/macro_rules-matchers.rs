// This is a regression test for issue #86208.
// It is also a general test of macro_rules! display.

#![crate_name = "foo"]

// @has 'foo/macro.todo.html'
// @has - '//span[@class="macro"]' 'macro_rules!'
// @hasraw - ' todo {'

// @hasraw - '{ () =&gt; { ... }; ($('
// @has - '//span[@class="macro-nonterminal"]' '$'
// @has - '//span[@class="macro-nonterminal"]' 'arg'
// @hasraw - ':tt)+'
// @hasraw - ') =&gt; { ... }; }'
pub use std::todo;

mod mod1 {
    // @has 'foo/macro.macro1.html'
    // @hasraw - 'macro_rules!'
    // @hasraw - 'macro1'
    // @hasraw - '{ () =&gt; { ... }; ($('
    // @has - '//span[@class="macro-nonterminal"]' '$'
    // @has - '//span[@class="macro-nonterminal"]' 'arg'
    // @hasraw - ':'
    // @hasraw - 'expr'
    // @hasraw - '),'
    // @hasraw - '+'
    // @hasraw - ') =&gt; { ... }; }'
    #[macro_export]
    macro_rules! macro1 {
        () => {};
        ($($arg:expr),+) => { stringify!($($arg),+) };
    }
}
