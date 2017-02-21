# Statements and expressions

Rust is _primarily_ an expression language. This means that most forms of
value-producing or effect-causing evaluation are directed by the uniform syntax
category of _expressions_. Each kind of expression can typically _nest_ within
each other kind of expression, and rules for evaluation of expressions involve
specifying both the value produced by the expression and the order in which its
sub-expressions are themselves evaluated.

In contrast, statements in Rust serve _mostly_ to contain and explicitly
sequence expression evaluation.
