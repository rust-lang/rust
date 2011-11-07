# Introduction

## Scope

This is a tutorial for the Rust programming language. It assumes the
reader is familiar with the basic concepts of programming, and has
programmed in one or more other languages before. The tutorial covers
the whole language, though not with the depth and precision of the
[language reference][1].

FIXME: maybe also the stdlib?

[1]: http://www.rust-lang.org/doc/rust.html

## Disclaimer

Rust is a language under development. The general flavor of the
language has settled, but details will continue to change as it is
further refined. Nothing in this tutorial is final, and though we try
to keep it updated, it is possible that the text occasionally does not
reflect the actual state of the language.

## First Impressions

Though syntax is something you get used to, an initial encounter with
a language can be made easier if the notation looks familiar. Rust is
a curly-brace language in the tradition of C, C++, and JavaScript.

    fn fac(n: int) -> int {
        let result = 1, i = 1;
        while i <= n {
            result *= i;
            i += 1;
        }
        ret result;
    }

Several differences from C stand out. Types do not come before, but
after variable names (preceded by a colon). In local variables
(introduced with `let`), they are optional, and will be inferred when
left off. Constructs like `while` and `if` do not require parenthesis
around the condition (though they allow them). Also, there's a
tendency towards aggressive abbreviation in the keywords—`fn` for
function, `ret` for return.

You should, however, not conclude that Rust is simply an evolution of
C. As will become clear in the rest of this tutorial, it goes into
quite a different direction.

## Conventions

Throughout the tutorial, words that indicate language keywords or
identifiers defined in the example code are displayed in `code font`.

Code snippets are indented, and also shown in a monospace font. Not
all snippets constitute whole programs. For brevity, we'll often show
fragments of programs that don't compile on their own. To try them
out, you'll have to wrap them in `fn main() { ... }`, and make sure
they don't contain references to things that aren't actually defined.
