% Rust Macros Tutorial

# Introduction

Functions are the primary tool that programmers can use to build
abstractions. Sometimes, though, programmers want to abstract over
compile-time, syntactic structures rather than runtime values. For example,
the following two code fragments both pattern-match on their input and return
early in one case, doing nothing otherwise:

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

This code could become tiresome if repeated many times. However, there is no
straightforward way to rewrite it without the repeated code, using functions
alone. There is a solution, though: defining a macro to solve the problem. Macros are
lightweight custom syntax extensions, themselves defined using the
`macro_rules!` syntax extension. The following `early_return` macro captures
the pattern in the above code:

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

Macros are defined in pattern-matching style: in the above example, the text
`($inp:expr $sp:ident)` that appears on the left-hand side of the `=>` is the
*macro invocation syntax*, a pattern denoting how to write a call to the
macro. The text on the right-hand side of the `=>`, beginning with `match
$inp`, is the *macro transcription syntax*: what the macro expands to.

# Invocation syntax

The macro invocation syntax specifies the syntax for the arguments to the
macro. It appears on the left-hand side of the `=>` in a macro definition. It
conforms to the following rules:

1. It must be surrounded by parentheses.
2. `$` has special meaning.
3. The `()`s, `[]`s, and `{}`s it contains must balance. For example, `([)` is
forbidden.

Otherwise, the invocation syntax is free-form.

To take as an argument a fragment of Rust code, write `$` followed by a name
 (for use on the right-hand side), followed by a `:`, followed by a *fragment
 specifier*. The fragment specifier denotes the sort of fragment to match. The
 most common fragment specifiers are:

* `ident` (an identifier, referring to a variable or item. Examples: `f`, `x`,
  `foo`.)
* `expr` (an expression. Examples: `2 + 2`; `if true then { 1 } else { 2 }`;
  `f(42)`.)
* `ty` (a type. Examples: `int`, `~[(char, ~str)]`, `&T`.)
* `pat` (a pattern, usually appearing in a `match` or on the left-hand side of
  a declaration. Examples: `Some(t)`; `(17, 'a')`; `_`.)
* `block` (a sequence of actions. Example: `{ log(error, "hi"); return 12; }`)
 
The parser interprets any token that's not preceded by a `$` literally. Rust's usual
rules of tokenization apply,

So `($x:ident -> (($e:expr)))`, though excessively fancy, would designate a macro
that could be invoked like: `my_macro!(i->(( 2+2 )))`.

# Transcription syntax

The right-hand side of the `=>` follows the same rules as the left-hand side,
except that a `$` need only be followed by the name of the syntactic fragment
to transcribe into the macro expansion; its type need not be repeated.

The right-hand side must be enclosed by delimiters, and must be
an expression. Currently, invocations of user-defined macros can only appear in a context
where the Rust grammar requires an expression, even though `macro_rules!` itself can appear
in a context where the grammar requires an item.

# Multiplicity

## Invocation

Going back to the motivating example, recall that `early_return` expanded into
a `match` that would `return` if the `match`'s scrutinee matched the
"special case" identifier provided as the second argument to `early_return`,
and do nothing otherwise. Now suppose that we wanted to write a
version of `early_return` that could handle a variable number of "special"
cases.

The syntax `$(...)*` on the left-hand side of the `=>` in a macro definition
accepts zero or more occurrences of its contents. It works much
like the `*` operator in regular expressions. It also supports a
separator token (a comma-separated list could be written `$(...),*`), and `+`
instead of `*` to mean "at least one".

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
side of a macro definition. The behavior of `*` in transcription,
especially in cases where multiple `*`s are nested, and multiple different
names are involved, can seem somewhat magical and intuitive at first. The
system that interprets them is called "Macro By Example". The two rules to
keep in mind are (1) the behavior of `$(...)*` is to walk through one "layer"
of repetitions for all of the `$name`s it contains in lockstep, and (2) each
`$name` must be under at least as many `$(...)*`s as it was matched against.
If it is under more, it'll be repeated, as appropriate.

## Parsing limitations


For technical reasons, there are two limitations to the treatment of syntax
fragments by the macro parser:

1. The parser will always parse as much as possible of a Rust syntactic
fragment. For example, if the comma were omitted from the syntax of
`early_return!` above, `input_1 [` would've been interpreted as the beginning
of an array index. In fact, invoking the macro would have been impossible.
2. The parser must have eliminated all ambiguity by the time it reaches a 
`$name:fragment_specifier` declaration. This limitation can result in parse
errors when declarations occur at the beginning of, or immediately after,
a `$(...)*`. For example, the grammar `$($t:ty)* $e:expr` will always fail to
parse because the parser would be forced to choose between parsing `t` and
parsing `e`. Changing the invocation syntax to require a distinctive token in
front can solve the problem. In the above example, `$(T $t:ty)* E $e:exp`
solves the problem.

## A final note

Macros, as currently implemented, are not for the faint of heart. Even
ordinary syntax errors can be more difficult to debug when they occur inside a
macro, and errors caused by parse problems in generated code can be very
tricky. Invoking the `log_syntax!` macro can help elucidate intermediate
states, invoking `trace_macros!(true)` will automatically print those
intermediate states out, and passing the flag `--pretty expanded` as a
command-line argument to the compiler will show the result of expansion.
