# Macros

Functions are the programmer's primary tool of abstraction, but there are
cases in which they are insufficient, because the programmer wants to
abstract over concepts not represented as values. Consider the following
example:

~~~~
# enum t { special_a(uint), special_b(uint) };
# fn f() -> uint {
# let input_1 = special_a(0), input_2 = special_a(0);
match input_1 {
    special_a(x) => { return x; }
    _ => {}
}
// ...
match input_2 {
    special_b(x) => { return x; }
    _ => {}
}
# return 0u;
# }
~~~~

This code could become tiresome if repeated many times. However, there is
no reasonable function that could be written to solve this problem. In such a
case, it's possible to define a macro to solve the problem. Macros are
lightweight custom syntax extensions, themselves defined using the
`macro_rules!` syntax extension:

~~~~
# enum t { special_a(uint), special_b(uint) };
# fn f() -> uint {
# let input_1 = special_a(0), input_2 = special_a(0);
macro_rules! early_return(
    ($inp:expr $sp:ident) => ( //invoke it like `(input_5 special_e)`
        match $inp {
            $sp(x) => { return x; }
            _ => {}
        }
    );
);
// ...
early_return!(input_1 special_a);
// ...
early_return!(input_2 special_b);
# return 0;
# }
~~~~

Macros are defined in pattern-matching style:

## Invocation syntax

On the left-hand-side of the `=>` is the macro invocation syntax. It is
free-form, excepting the following rules:

1. It must be surrounded in parentheses.
2. `$` has special meaning.
3. The `()`s, `[]`s, and `{}`s it contains must balance. For example, `([)` is
forbidden.

To take as an argument a fragment of Rust code, write `$` followed by a name
 (for use on the right-hand side), followed by a `:`, followed by the sort of
fragment to match (the most common ones are `ident`, `expr`, `ty`, `pat`, and
`block`). Anything not preceeded by a `$` is taken literally. The standard
rules of tokenization apply,

So `($x:ident => (($e:expr)))`, though excessively fancy, would create a macro
that could be invoked like `my_macro!(i=>(( 2+2 )))`.

## Transcription syntax

The right-hand side of the `=>` follows the same rules as the left-hand side,
except that `$` need only be followed by the name of the syntactic fragment
to transcribe.

The right-hand side must be surrounded by delimiters of some kind, and must be
an expression; currently, user-defined macros can only be invoked in
expression position (even though `macro_rules!` itself can be in item
position).

## Multiplicity

### Invocation

Going back to the motivating example, suppose that we wanted each invocation
of `early_return` to potentially accept multiple "special" identifiers. The
syntax `$(...)*` accepts zero or more occurences of its contents, much like
the Kleene star operator in regular expressions. It also supports a separator
token (a comma-separated list could be written `$(...),*`), and `+` instead of
`*` to mean "at least one".

~~~~
# enum t { special_a(uint),special_b(uint),special_c(uint),special_d(uint)};
# fn f() -> uint {
# let input_1 = special_a(0), input_2 = special_a(0);
macro_rules! early_return(
    ($inp:expr, [ $($sp:ident)|+ ]) => (
        match $inp {
            $(
                $sp(x) => { return x; }
            )+
            _ => {}
        }
    );
);
// ...
early_return!(input_1, [special_a|special_c|special_d]);
// ...
early_return!(input_2, [special_b]);
# return 0;
# }
~~~~

### Transcription

As the above example demonstrates, `$(...)*` is also valid on the right-hand
side of a macro definition. The behavior of Kleene star in transcription,
especially in cases where multiple stars are nested, and multiple different
names are involved, can seem somewhat magical and intuitive at first. The
system that interprets them is called "Macro By Example". The two rules to
keep in mind are (1) the behavior of `$(...)*` is to walk through one "layer"
of repetitions for all of the `$name`s it contains in lockstep, and (2) each
`$name` must be under at least as many `$(...)*`s as it was matched against.
If it is under more, it'll will be repeated, as appropriate.

## Parsing limitations

The parser used by the macro system is reasonably powerful, but the parsing of
Rust syntax is restricted in two ways:

1. The parser will always parse as much as possible. For example, if the comma
were omitted from the syntax of `early_return!` above, `input_1 [` would've
been interpreted as the beginning of an array index. In fact, invoking the
macro would have been impossible.
2. The parser must have eliminated all ambiguity by the time it reaches a
`$name:fragment_specifier`. This most often affects them when they occur in
the beginning of, or immediately after, a `$(...)*`; requiring a distinctive
token in front can solve the problem.

## A final note

Macros, as currently implemented, are not for the faint of heart. Even
ordinary syntax errors can be more difficult to debug when they occur inside
a macro, and errors caused by parse problems in generated code can be very
tricky. Invoking the `log_syntax!` macro can help elucidate intermediate
states, using `trace_macros!(true)` will automatically print those
intermediate states out, and using `--pretty expanded` as an argument to the
compiler will show the result of expansion.


