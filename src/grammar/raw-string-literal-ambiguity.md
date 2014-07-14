Rust's lexical grammar is not context-free. Raw string literals are the source
of the problem. Informally, a raw string literal is an `r`, followed by `N`
hashes (where N can be zero), a quote, any characters, then a quote followed
by `N` hashes. This grammar describes this as best possible:

    R -> 'r' S
    S -> '"' B '"'
    S -> '#' S '#'
    B -> . B
    B -> ε

Where `.` represents any character, and `ε` the empty string. Consider the
string `r#""#"#`. This string is not a valid raw string literal, but can be
accepted as one by the above grammar, using the derivation:

    R : #""#"#
    S : ""#"
    S : "#
    B : #
    B : ε

(Where `T : U` means the rule `T` is applied, and `U` is the remainder of the
string.) The difficulty arises from the fact that it is fundamentally
context-sensitive. In particular, the context needed is the number of hashes.
I know of no way to resolve this, but also have not come up with a proof that
it is not context sensitive. Such a proof would probably use the pumping lemma
for context-free languages, but I (cmr) could not come up with a proof after
spending a few hours on it, and decided my time best spent elsewhere. Pull
request welcome!
