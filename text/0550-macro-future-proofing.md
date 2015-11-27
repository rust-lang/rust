- Start Date: 2014-12-21
- RFC PR: [550](https://github.com/rust-lang/rfcs/pull/550)
- Rust Issue: [20563](https://github.com/rust-lang/rust/pull/20563)

# Summary

Future-proof the allowed forms that input to an MBE can take by requiring
certain delimiters following NTs in a matcher. In the future, it will be
possible to lift these restrictions backwards compatibly if desired.

# Key Terminology

- `macro`: anything invokable as `foo!(...)` in source code.
- `MBE`: macro-by-example, a macro defined by `macro_rules`.
- `matcher`: the left-hand-side of a rule in a `macro_rules` invocation, or a subportion thereof.
- `macro parser`: the bit of code in the Rust parser that will parse the input using a grammar derived from all of the matchers.
- `fragment`: The class of Rust syntax that a given matcher will accept (or "match").
- `repetition` : a fragment that follows a regular repeating pattern
- `NT`: non-terminal, the various "meta-variables" or repetition matchers that can appear in a matcher, specified in MBE syntax with a leading `$` character.
- `simple NT`: a "meta-variable" non-terminal (further discussion below).
- `complex NT`: a repetition matching non-terminal, specified via Kleene closure operators (`*`, `+`).
- `token`: an atomic element of a matcher; i.e. identifiers, operators, open/close delimiters, *and* simple NT's.
- `token tree`: a tree structure formed from tokens (the leaves), complex NT's, and finite sequences of token trees.
- `delimiter token`: a token that is meant to divide the end of one fragment and the start of the next fragment.
- `separator token`: an optional delimiter token in an complex NT that separates each pair of elements in the matched repetition.
- `separated complex NT`: a complex NT that has its own separator token.
- `delimited sequence`: a sequence of token trees with appropriate open- and close-delimiters at the start and end of the sequence.
- `empty fragment`: The class of invisible Rust syntax that separates tokens, i.e. whitespace, or (in some lexical contexts), the empty token sequence.
- `fragment specifier`: The identifier in a simple NT that specifies which fragment the NT accepts.
- `language`: a context-free language.

Example:

```rust
macro_rules! i_am_an_mbe {
    (start $foo:expr $($i:ident),* end) => ($foo)
}
```

`(start $foo:expr $($i:ident),* end)` is a matcher. The whole matcher
is a delimited sequence (with open- and close-delimiters `(` and `)`),
and `$foo` and `$i` are simple NT's with `expr` and `ident` as their
respective fragment specifiers.

`$(i:ident),*` is *also* an NT; it is a complex NT that matches a
comma-seprated repetition of identifiers. The `,` is the separator
token for the complex NT; it occurs in between each pair of elements
(if any) of the matched fragment.

Another example of a complex NT is `$(hi $e:expr ;)+`, which matches
any fragment of the form `hi <expr>; hi <expr>; ...` where `hi
<expr>;` occurs at least once. Note that this complex NT does not
have a dedicated separator token.

(Note that Rust's parser ensures that delimited sequences always occur
with proper nesting of token tree structure and correct matching of open-
and close-delimiters.)

# Motivation

In current Rust (version 0.12; i.e. pre 1.0), the `macro_rules` parser is very liberal in what it accepts
in a matcher. This can cause problems, because it is possible to write an
MBE which corresponds to an ambiguous grammar. When an MBE is invoked, if the
macro parser encounters an ambiguity while parsing, it will bail out with a
"local ambiguity" error. As an example for this, take the following MBE:

```rust
macro_rules! foo {
    ($($foo:expr)* $bar:block) => (/*...*/)
};
```

Attempts to invoke this MBE will never succeed, because the macro parser
will always emit an ambiguity error rather than make a choice when presented
an ambiguity. In particular, it needs to decide when to stop accepting
expressions for `foo` and look for a block for `bar` (noting that blocks are
valid expressions). Situations like this are inherent to the macro system. On
the other hand, it's possible to write an unambiguous matcher that becomes
ambiguous due to changes in the syntax for the various fragments. As a
concrete example:

```rust
macro_rules! bar {
    ($in:ty ( $($arg:ident)*, ) -> $out:ty;) => (/*...*/)
};
```

When the type syntax was extended to include the unboxed closure traits,
an input such as `FnMut(i8, u8) -> i8;` became ambiguous. The goal of this
proposal is to prevent such scenarios in the future by requiring certain
"delimiter tokens" after an NT. When extending Rust's syntax in the future,
ambiguity need only be considered when combined with these sets of delimiters,
rather than any possible arbitrary matcher.

----

Another example of a potential extension to the language that
motivates a restricted set of "delimiter tokens" is
([postponed][Postponed 961]) [RFC 352][], "Allow loops to return
values other than `()`", where the `break` expression would now accept
an optional input expression: `break <expr>`.

 * This proposed extension to the language, combined with the facts that
   `break` and `{ <stmt> ... <expr>? }` are Rust expressions, implies that
   `{` should not be in the follow set for the `expr` fragment specifier.

 * Thus in a slightly more ideal world the following program would not be
   accepted, because the interpretation of the macro could change if we
   were to accept RFC 352:

   ```rust
   macro_rules! foo {
       ($e:expr { stuff }) => { println!("{:?}", $e) }
   }

   fn main() {
       loop { foo!(break { stuff }); }
   }
   ```

   (in our non-ideal world, the program is legal in Rust versions 1.0
   through at least 1.4)

[RFC 352]: https://github.com/rust-lang/rfcs/pull/352

[Postponed 961]: https://github.com/rust-lang/rfcs/issues/961

# Detailed design

We will tend to use the variable "M" to stand for a matcher,
variables "t" and "u" for arbitrary individual tokens,
and the variables "tt" and "uu" for arbitrary token trees.
(The use of "tt" does present potential ambiguity with its
additional role as a fragment specifier; but it will be clear
from context which interpretation is meant.)

"SEP" will range over separator tokens,
"OP" over the Kleene operators `*` and `+`, and
"OPEN"/"CLOSE" over matching token pairs surrounding a delimited sequence (e.g. `[` and `]`).

We also use Greek letters "α" "β" "γ" "δ" to stand for potentially empty
token-tree sequences. (However, the
Greek letter "ε" (epsilon) has a special role in the presentation and
does not stand for a token-tree sequence.)

 * This Greek letter convention is usually just employed when the
   presence of a sequence is a technical detail; in particular, when I
   wish to *emphasize* that we are operating on a sequence of
   token-trees, I will use the notation "tt ..." for the sequence, not
   a Greek letter

Note that a matcher is merely a token tree. A "simple NT", as
mentioned above, is an meta-variable NT; thus it is a
non-repetition. For example, `$foo:ty` is a simple NT but
`$($foo:ty)+` is a complex NT.

Note also that in the context of this RFC, the term "token" generally
*includes* simple NTs.

Finally, it is useful for the reader to keep in mind that according to
the definitions of this RFC, no simple NT matches
the empty fragment, and likewise no token matches
the empty fragment of Rust syntax. (Thus, the *only* NT that can match
the empty fragment is a complex NT.)

## The Matcher Invariant

This RFC establishes the following two-part invariant for valid matchers

 1. For any two successive token tree sequences in a matcher `M`
    (i.e. `M = ... tt uu ...`), we must have
    FOLLOW(`... tt`) ⊇ FIRST(`uu ...`)

 2. For any separated complex NT in a matcher, `M = ... $(tt ...) SEP OP ...`,
    we must have
    `SEP` ∈ FOLLOW(`tt ...`).

The first part says that whatever actual token that comes after a
matcher must be somewhere in the predetermined follow set.  This
ensures that a legal macro definition will continue to assign the same
determination as to where `... tt` ends and `uu ...` begins, even as
new syntactic forms are added to the language.

The second part says that a separated complex NT must use a seperator
token that is part of the predetermined follow set for the internal
contents of the NT. This ensures that a legal macro definition will
continue to parse an input fragment into the same delimited sequence
of `tt ...`'s, even as new syntactic forms are added to the language.

(This is assuming that all such changes are appropriately restricted,
by the definition of FOLLOW below, of course.)

The above invariant is only formally meaningful if one knows what
FIRST and FOLLOW denote. We address this in the following sections.

## FIRST and FOLLOW, informally

FIRST and FOLLOW are defined as follows.

A given matcher M maps to three sets: FIRST(M), LAST(M) and FOLLOW(M).

Each of the three sets is made up of tokens. FIRST(M) and LAST(M) may
also contain a distinguished non-token element ε ("epsilon"), which
indicates that M can match the empty fragment. (But FOLLOW(M) is
always just a set of tokens.)

Informally:

 * FIRST(M): collects the tokens potentially used first when matching a fragment to M.

 * LAST(M): collects the tokens potentially used last when matching a fragment to M.

 * FOLLOW(M): the set of tokens allowed to follow immediately after some fragment
   matched by M.

   In other words: t ∈ FOLLOW(M) if and only if there exists (potentially empty) token sequences α, β, γ, δ where:
   * M matches β,
   * t matches γ, and
   * The concatenation α β γ δ is a parseable Rust program.

We use the shorthand ANYTOKEN to denote the set of all tokens (including simple NTs).

 * (For example, if any token is legal after a matcher M, then FOLLOW(M) = ANYTOKEN.)

(To review one's understanding of the above informal descriptions, the
reader at this point may want to jump ahead to the
[examples of FIRST/LAST][examples-of-first-and-last] before reading
their formal definitions.)

## FIRST, LAST

Below are formal inductive definitions for FIRST and LAST.

"A ∪ B" denotes set union, "A ∩ B" denotes set intersection, and
"A \ B" denotes set difference (i.e. all elements of A that are not present
in B).

FIRST(M), defined by case analysis on the sequence M and the structure
of its first token-tree (if any):

  * if M is the empty sequence, then FIRST(M) = { ε },

  * if M starts with a token t, then FIRST(M) = { t },

    (Note: this covers the case where M starts with a delimited
    token-tree sequence, `M = OPEN tt ... CLOSE ...`, in which case `t = OPEN` and
    thus FIRST(M) = { `OPEN` }.)

    (Note: this critically relies on the property that no simple NT matches the
     empty fragment.)

  * Otherwise, M is a token-tree sequence starting with a complex NT:
    `M = $( tt ... ) OP α`, or `M = $( tt ... ) SEP OP α`,
    (where `α` is the (potentially empty) sequence of token trees for the rest of the matcher).

    * Let sep_set = { SEP } if SEP present; otherwise sep_set = {}.

    * If ε ∈ FIRST(`tt ...`), then FIRST(M) = (FIRST(`tt ...`) \ { ε }) ∪ sep_set ∪ FIRST(`α`)

    * Else if OP = `*`, then FIRST(M) = FIRST(`tt ...`) ∪ FIRST(`α`)

    * Otherwise (OP = `+`), FIRST(M) = FIRST(`tt ...`)

Note: The ε-case above,

> FIRST(M) = (FIRST(`tt ...`) \ { ε }) ∪ sep_set ∪ FIRST(`α`)

may seem complicated, so lets take a moment to break it down. In the
ε case, the sequence `tt ...` may be empty. Therefore our first
token may be `SEP` itself (if it is present), or it may be the first
token of `α`); that's why the result is including "sep_set ∪
FIRST(`α`)". Note also that if `α` itself may match the empty
fragment, then FIRST(`α`) will ensure that ε is included in our
result, and conversely, if `α` cannot match the empty fragment, then
we must *ensure* that ε is *not* included in our result; these two
facts together are why we can and should unconditionally remove ε
from FIRST(`tt ...`).

----

LAST(M), defined by case analysis on M itself (a sequence of token-trees):

  * if M is the empty sequence, then LAST(M) = { ε }

  * if M is a singleton token t, then LAST(M) = { t }

  * if M is the singleton complex NT repeating zero or more times,
    `M = $( tt ... ) *`, or `M = $( tt ... ) SEP *`

    * Let sep_set = { SEP } if SEP present; otherwise sep_set = {}.

    * if ε ∈ LAST(`tt ...`) then LAST(M) = LAST(`tt ...`) ∪ sep_set

    * otherwise, the sequence `tt ...` must be non-empty; LAST(M) = LAST(`tt ...`) ∪ { ε }

  * if M is the singleton complex NT repeating one or more times,
     `M = $( tt ... ) +`, or `M = $( tt ... ) SEP +`

    * Let sep_set = { SEP } if SEP present; otherwise sep_set = {}.

    * if ε ∈ LAST(`tt ...`) then LAST(M) = LAST(`tt ...`) ∪ sep_set

    * otherwise, the sequence `tt ...` must be non-empty; LAST(M) = LAST(`tt ...`)

  * if M is a delimited token-tree sequence `OPEN tt ... CLOSE`, then LAST(M) = { `CLOSE` }

  * if M is a non-empty sequence of token-trees `tt uu ...`,

    * If ε ∈ LAST(`uu ...`), then LAST(M) = LAST(`tt`) ∪ (LAST(`uu ...`) \ { ε }).

    * Otherwise, the sequence `uu ...` must be non-empty; then LAST(M) = LAST(`uu ...`)

NOTE: The presence or absence of SEP *is* relevant to the above
definitions, but solely in the case where the interior of the complex
NT could be empty (i.e. ε ∈ FIRST(interior)). (I overlooked this fact
in my first round of prototyping.)

NOTE: The above definition for LAST assumes that we keep our
pre-existing rule that the seperator token in a complex NT is *solely* for
separating elements; i.e. that such NT's do not match fragments that
*end with* the seperator token. If we choose to lift this restriction
in the future, the above definition will need to be revised
accordingly.

## Examples of FIRST and LAST
[examples-of-first-and-last]: #examples-of-first-and-last

Below are some examples of FIRST and LAST.
(Note in particular how the special ε element is introduced and
eliminated based on the interation between the pieces of the input.)

Our first example is presented in a tree structure to elaborate on how
the analysis of the matcher composes. (Some of the simpler subtrees
have been elided.)

    INPUT:  $(  $d:ident   $e:expr   );*    $( $( h )* );*    $( f ; )+   g
                ~~~~~~~~   ~~~~~~~                ~
                    |         |                   |
    FIRST:   { $d:ident }  { $e:expr }          { h }


    INPUT:  $(  $d:ident   $e:expr   );*    $( $( h )* );*    $( f ; )+
                ~~~~~~~~~~~~~~~~~~             ~~~~~~~           ~~~
                           |                      |               |
    FIRST:          { $d:ident }               { h, ε }         { f }

    INPUT:  $(  $d:ident   $e:expr   );*    $( $( h )* );*    $( f ; )+   g
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~    ~~~~~~~~~~~~~~    ~~~~~~~~~   ~
                           |                       |              |       |
    FIRST:        { $d:ident, ε }            {  h, ε, ;  }      { f }   { g }


    INPUT:  $(  $d:ident   $e:expr   );*    $( $( h )* );*    $( f ; )+   g
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                            |
    FIRST:                       { $d:ident, h, ;,  f }

Thus:

 * FIRST(`$($d:ident $e:expr );* $( $(h)* );* $( f ;)+ g`) = { `$d:ident`, `h`, `;`, `f` }

Note however that:

 * FIRST(`$($d:ident $e:expr );* $( $(h)* );* $($( f ;)+ g)*`) = { `$d:ident`, `h`, `;`, `f`, ε }

Here are similar examples but now for LAST.

 * LAST(`$d:ident $e:expr`) = { `$e:expr` }
 * LAST(`$( $d:ident $e:expr );*`) = { `$e:expr`, ε }
 * LAST(`$( $d:ident $e:expr );* $(h)*`) = { `$e:expr`, ε, `h` }
 * LAST(`$( $d:ident $e:expr );* $(h)* $( f ;)+`) = { `;` }
 * LAST(`$( $d:ident $e:expr );* $(h)* $( f ;)+ g`) = { `g` }
 
 and again, changing the end part of matcher changes its last set considerably:
 
 * LAST(`$( $d:ident $e:expr );* $(h)* $($( f ;)+ g)*`) = { `$e:expr`, ε, `h`, `g` }

## FOLLOW(M)

Finally, the definition for `FOLLOW(M)` is built up incrementally atop
more primitive functions.

We first assume a primitive mapping, `FOLLOW(NT)` (defined
[below][follow-nt]) from a simple NT to the set of allowed tokens for
the fragment specifier for that NT.

Second, we generalize FOLLOW to tokens: FOLLOW(t) = FOLLOW(NT) if t is (a simple) NT.
Otherwise, t must be some other (non NT) token; in this case FOLLOW(t) = ANYTOKEN.

Finally, we generalize FOLLOW to arbitrary matchers by composing the primitive
functions above:

```
FOLLOW(M) = FOLLOW(t1) ∩ FOLLOW(t2) ∩ ... ∩ FOLLOW(tN)
            where { t1, t2, ..., tN } = (LAST(M) \ { ε })
```

Examples of FOLLOW (expressed as equality relations between sets, to avoid
incoporating details of FOLLOW(NT) in these examples):

 * FOLLOW(`$( $d:ident $e:expr )*`) = FOLLOW(`$e:expr`)
 * FOLLOW(`$( $d:ident $e:expr )* $(;)*`) = FOLLOW(`$e:expr`) ∩ ANYTOKEN = FOLLOW(`$e:expr`)
 * FOLLOW(`$( $d:ident $e:expr )* $(;)* $( f |)+`) = ANYTOKEN

## FOLLOW(NT)
[follow-nt]: #follownt

Here is the definition for FOLLOW(NT), which maps every simple NT to
the set of tokens that are allowed to follow it, based on the fragment
specifier for the NT.

The current legal fragment specifiers are: `item`, `block`, `stmt`, `pat`,
`expr`, `ty`, `ident`, `path`, `meta`, and `tt`.

- `FOLLOW(pat)` = `{FatArrow, Comma, Eq, Or}`
- `FOLLOW(expr)` = `{FatArrow, Comma, Semicolon}`
- `FOLLOW(ty)` = `{OpenDelim(Brace), Comma, FatArrow, Colon, Eq, Gt, Ident(as), Ident(where), Semi, Or}`
- `FOLLOW(stmt)` = `FOLLOW(expr)`
- `FOLLOW(path)` = `FOLLOW(ty)`
- `FOLLOW(block)` = any token
- `FOLLOW(ident)` = any token
- `FOLLOW(tt)` = any token
- `FOLLOW(item)` = any token
- `FOLLOW(meta)` = any token

(Note that close delimiters are valid following any NT.)

## Examples of valid and invalid matchers

With the above specification in hand, we can present arguments for
why particular matchers are legal and others are not.

 * `($ty:ty < foo ,)` : illegal, because FIRST(`< foo ,`) = { `<` } ⊈ FOLLOW(`ty`)

 * `($ty:ty , foo <)` :   legal, because FIRST(`, foo <`) = { `,` }  is ⊆ FOLLOW(`ty`).

 * `($pa:pat $pb:pat $ty:ty ,)` : illegal, because FIRST(`$pb:pat $ty:ty ,`) = { `$pb:pat` } ⊈ FOLLOW(`pat`), and also FIRST(`$ty:ty ,`) = { `$ty:ty` } ⊈ FOLLOW(`pat`).

 * `( $($a:tt $b:tt)* ; )` : legal, because FIRST(`$b:tt`) = { `$b:tt` } is ⊆ FOLLOW(`tt`) = ANYTOKEN, as is FIRST(`;`) = { `;` }.

 * `( $($t:tt),* , $(t:tt),* )` : legal (though any attempt to actually use this macro will signal a local ambguity error during expansion).

 * `($ty:ty $(; not sep)* -)` : illegal, because FIRST(`$(; not sep)* -`) = { `;`, `-` } is not in FOLLOW(`ty`).

 * `($($ty:ty)-+)` : illegal, because separator `-` is not in FOLLOW(`ty`).


# Drawbacks

It does restrict the input to a MBE, but the choice of delimiters provides
reasonable freedom and can be extended in the future.

# Alternatives

1. Fix the syntax that a fragment can parse. This would create a situation
   where a future MBE might not be able to accept certain inputs because the
   input uses newer features than the fragment that was fixed at 1.0. For
   example, in the `bar` MBE above, if the `ty` fragment was fixed before the
   unboxed closure sugar was introduced, the MBE would not be able to accept
   such a type. While this approach is feasible, it would cause unnecessary
   confusion for future users of MBEs when they can't put certain perfectly
   valid Rust code in the input to an MBE. Versioned fragments could avoid
   this problem but only for new code.
2. Keep `macro_rules` unstable. Given the great syntactical abstraction that
   `macro_rules` provides, it would be a shame for it to be unusable in a
   release version of Rust. If ever `macro_rules` were to be stabilized, this
   same issue would come up.
3. Do nothing. This is very dangerous, and has the potential to essentially
   freeze Rust's syntax for fear of accidentally breaking a macro.

# Edit History

- Updated by https://github.com/rust-lang/rfcs/pull/1209, which added
  semicolons into the follow set for types.

- Updated by https://github.com/rust-lang/rfcs/pull/1384:
  * replaced detailed design with a specification-oriented presentation rather than an implementation-oriented algorithm.
  * fixed some oversights in the specification (that led to matchers like `break { stuff }` being accepted),
  * expanded the follows sets for `ty` to include `OpenDelim(Brace), Ident(where), Or` (since Rust's grammar already requires all of `|foo:TY| {}`, `fn foo() -> TY {}` and `fn foo() -> TY where {}` to work).
  * expanded the follow set for `pat` to include `Or` (since Rust's grammar already requires `match (true,false) { PAT | PAT => {} }` and `|PAT| {}` to work). See also [RFC issue 1336][].

[RFC issue 1336]: https://github.com/rust-lang/rfcs/issues/1336

# Appendices

## Appendix A: Algorithm for recognizing valid matchers.

The detailed design above only sought to provide a *specification* for
what a correct matcher is (by defining FIRST, LAST, and FOLLOW, and
specifying the invariant relating FIRST and FOLLOW for all valid
matchers.

The above specification can be implemented efficiently; we here give
one example algorithm for recognizing valid matchers.

 * This is not the only possible algorithm; for example, one could
   precompute a table mapping every suffix of every token-tree
   sequence to its FIRST set, by augmenting `FirstSet` below
   accordingly.

   Or one could store a subset of such information during the
   precomputation, such as just the FIRST sets for complex NT's, and
   then use that table to inform a *forward scan* of the input.

   The latter is in fact what my prototype implementation does; I must
   emphasize the point that the algorithm here is not prescriptive.

 * The intent of this RFC is that the specifications of FIRST
   and FOLLOW above will take precedence over this algorithm if the two
   are found to be producing inconsistent results.

The algorithm for recognizing valid matchers `M` is named ValidMatcher.

To define it, we will need a mapping from submatchers of M to the
FIRST set for that submatcher; that is handled by `FirstSet`.

### Procedure FirstSet(M)

*input*: a token tree `M` representing a matcher

*output*: `FIRST(M)`

```
Let M = tts[1] tts[2] ... tts[n].
Let curr_first = { ε }.

For i in n down to 1 (inclusive):
  Let tt = tts[i].

  1. If tt is a token, curr_first := { tt }

  2. Else if tt is a delimited sequence `OPEN uu ... ClOSE`,
     curr_first := { OPEN }

  3. Else tt is a complex NT `$(uu ...) SEP OP`

     Let inner_first = FirstSet(`uu ...`) i.e. recursive call

     if OP == `*` or ε ∈ inner_first then
         curr_first := curr_first ∪ inner_first
     else
         curr_first := inner_first

return curr_first
```

(Note: If we were precomputing a full table in this procedure, we would need
a recursive invocation on (uu ...) in step 2 of the for-loop.)

### Predicate ValidMatcher(M)

To simplify the specification, we assume in this presentation that all
simple NT's have a valid fragment specifier (i.e., one that has an
entry in the FOLLOW(NT) table above.

This algorithm works by scanning forward across the matcher M = α β,
(where α is the prefix we have scanned so far, and β is the suffix
that remains to be scanned). We maintain LAST(α) as we scan, and use
it to compute FOLLOW(α) and compare that to FIRST(β).

*input*: a token tree, `M`, and a set of tokens that could follow it, `F`.

*output*: LAST(M) (and also signals failure whenever M is invalid)

```
Let last_of_prefix = { ε }

Let M = tts[1] tts[2] ... tts[n].

For i in 1 up to n (inclusive):
  // For reference:
  // α = tts[1] .. tts[i]
  // β = tts[i+1] .. tts[n]
  // γ is some outer token sequence; the input F represents FIRST(γ)

  1. Let tt = tts[i].

  2. Let first_of_suffix; // aka FIRST(β γ)

  3. let S = FirstSet(tts[i+1] .. tts[n]);

  4. if ε ∈ S then
     // (include the follow information if necessary)

     first_of_suffix := S ∪ F

  5. else

     first_of_suffix := S

  6. Update last_of_prefix via case analysis on tt:

     a. If tt is a token:
        last_of_prefix := { tt }

     b. Else if tt is a delimited sequence `OPEN uu ... CLOSE`:

        i.  run ValidMatcher( M = `uu ...`, F = { `CLOSE` })

       ii. last_of_prefix := { `CLOSE` }

     c. Else, tt must be a complex NT,
        in other words, `NT = $( uu .. ) SEP OP` or `NT = $( uu .. ) OP`:

        i. If SEP present,
          let sublast = ValidMatcher( M = `uu ...`, F = first_of_suffix ∪ { `SEP` })

       ii. else:
          let sublast = ValidMatcher( M = `uu ...`, F = first_of_suffix)

      iii. If ε in sublast then:
           last_of_prefix := last_of_prefix ∪ (sublast \ ε)

       iv. Else:
           last_of_prefix := sublast

  7. At this point, last_of_prefix == LAST(α) and first_of_suffix == FIRST(β γ).

     For each simple NT token t in last_of_prefix:

     a. If first_of_suffix ⊆ FOLLOW(t), then we are okay so far. </li>

     b. Otherwise, we have found a token t whose follow set is not compatible
        with the FIRST(β γ), and must signal failure.

// After running the above for loop on all of `M`, last_of_prefix == LAST(M)

Return last_of_prefix
```

This algorithm should be run on every matcher in every `macro_rules`
invocation, with `F` = { `EOF` }. If it rejects a matcher, an error
should be emitted and compilation should not complete.
