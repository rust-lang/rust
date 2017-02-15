# String table productions

Some rules in the grammar &mdash; notably [unary
operators](#unary-operator-expressions), [binary
operators](#binary-operator-expressions), and [keywords][keywords] &mdash; are
given in a simplified form: as a listing of a table of unquoted, printable
whitespace-separated strings. These cases form a subset of the rules regarding
the [token](#tokens) rule, and are assumed to be the result of a
lexical-analysis phase feeding the parser, driven by a DFA, operating over the
disjunction of all such string table entries.

[keywords]: grammar.html#keywords

When such a string enclosed in double-quotes (`"`) occurs inside the grammar,
it is an implicit reference to a single member of such a string table
production. See [tokens](#tokens) for more information.
