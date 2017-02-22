# Macros By Example

`macro_rules` allows users to define syntax extension in a declarative way.  We
call such extensions "macros by example" or simply "macros".

Currently, macros can expand to expressions, statements, items, or patterns.

(A `sep_token` is any token other than `*` and `+`. A `non_special_token` is
any token other than a delimiter or `$`.)

The macro expander looks up macro invocations by name, and tries each macro
rule in turn. It transcribes the first successful match. Matching and
transcription are closely related to each other, and we will describe them
together.

The macro expander matches and transcribes every token that does not begin with
a `$` literally, including delimiters. For parsing reasons, delimiters must be
balanced, but they are otherwise not special.

In the matcher, `$` _name_ `:` _designator_ matches the nonterminal in the Rust
syntax named by _designator_. Valid designators are:

* `item`: an [item]
* `block`: a [block]
* `stmt`: a [statement]
* `pat`: a [pattern]
* `expr`: an [expression]
* `ty`: a [type]
* `ident`: an [identifier]
* `path`: a [path]
* `tt`: a token tree (a single [token] by matching `()`, `[]`, or `{}`)
* `meta`: the contents of an [attribute]

[item]: items.html
[block]: expressions.html#block-expressions
[statement]: statements.html
[pattern]: expressions.html#match-expressions
[expression]: expressions.html
[type]: types.html
[identifier]: identifiers.html
[path]: paths.html
[token]: tokens.html
[attribute]: attributes.html

In the transcriber, the
designator is already known, and so only the name of a matched nonterminal comes
after the dollar sign.

In both the matcher and transcriber, the Kleene star-like operator indicates
repetition. The Kleene star operator consists of `$` and parentheses, optionally
followed by a separator token, followed by `*` or `+`. `*` means zero or more
repetitions, `+` means at least one repetition. The parentheses are not matched or
transcribed. On the matcher side, a name is bound to _all_ of the names it
matches, in a structure that mimics the structure of the repetition encountered
on a successful match. The job of the transcriber is to sort that structure
out.

The rules for transcription of these repetitions are called "Macro By Example".
Essentially, one "layer" of repetition is discharged at a time, and all of them
must be discharged by the time a name is transcribed. Therefore, `( $( $i:ident
),* ) => ( $i )` is an invalid macro, but `( $( $i:ident ),* ) => ( $( $i:ident
),*  )` is acceptable (if trivial).

When Macro By Example encounters a repetition, it examines all of the `$`
_name_ s that occur in its body. At the "current layer", they all must repeat
the same number of times, so ` ( $( $i:ident ),* ; $( $j:ident ),* ) => ( $(
($i,$j) ),* )` is valid if given the argument `(a,b,c ; d,e,f)`, but not
`(a,b,c ; d,e)`. The repetition walks through the choices at that layer in
lockstep, so the former input transcribes to `(a,d), (b,e), (c,f)`.

Nested repetitions are allowed.

### Parsing limitations

The parser used by the macro system is reasonably powerful, but the parsing of
Rust syntax is restricted in two ways:

1. Macro definitions are required to include suitable separators after parsing
   expressions and other bits of the Rust grammar. This implies that
   a macro definition like `$i:expr [ , ]` is not legal, because `[` could be part
   of an expression. A macro definition like `$i:expr,` or `$i:expr;` would be legal,
   however, because `,` and `;` are legal separators. See [RFC 550] for more information.
2. The parser must have eliminated all ambiguity by the time it reaches a `$`
   _name_ `:` _designator_. This requirement most often affects name-designator
   pairs when they occur at the beginning of, or immediately after, a `$(...)*`;
   requiring a distinctive token in front can solve the problem.

[RFC 550]: https://github.com/rust-lang/rfcs/blob/master/text/0550-macro-future-proofing.md
