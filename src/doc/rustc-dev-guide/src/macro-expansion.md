# Macro expansion

Macro expansion happens during parsing. `rustc` has two parsers, in fact: the
normal Rust parser, and the macro parser. During the parsing phase, the normal
Rust parser will call into the macro parser when it encounters a macro. The
macro parser, in turn, may call back out to the Rust parser when it needs to
bind a metavariable (e.g. `$expr`). There are a few aspects of this system to be
explained. The code for macro expansion is in `src/libsyntax/ext/tt/`.

TODO: explain parsing of macro definitions

TODO: explain parsing of macro invokations + macro expansion
