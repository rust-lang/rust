% Glossary

Not every Rustacean has a background in systems programming, nor in computer
science, so we've added explanations of terms that might be unfamiliar.

### Arity

Arity refers to the number of arguments a function or operation takes.

```rust
let x = (2, 3);
let y = (4, 6);
let z = (8, 2, 6);
```

In the example above `x` and `y` have arity 2. `z` has arity 3.

### Abstract Syntax Tree

When a compiler is compiling your program, it does a number of different
things. One of the things that it does is turn the text of your program into an
‘abstract syntax tree’, or‘AST’. This tree is a representation of the
structure of your program. For example, `2 + 3` can be turned into a tree:

```text
  +
 / \
2   3
```

And `2 + (3 * 4)` would look like this:

```text
  +
 / \
2   *
   / \
  3   4
```
