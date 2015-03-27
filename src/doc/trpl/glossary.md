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
'abstract syntax tree,' or 'AST.' This tree is a representation of the
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

### Algebraic data types

Algebraic data type (ADT) is a mathematical type category which includes product
types (tuples) and sum types (enums) (also called tagged unions).

#### Product types

Consider product types: tuples. In tuples, the type is defined by the subtypes
and their order. That is:

```rust
let tuple1 = (7i32, 8i64); // (i32, i64) is a one type.
let tuple2 = (8i64, 7i32); // (i64, i32) is a different one.
```

Here, the subtypes were the same but the order was different; this makes them
different types. The product type name comes from the fact that a Cartesian
Product is a mathematical operation which given multiple types, will directly
compute all possible variants. For example, given `i32` and `i64` as two variants,
all possible variants from `i32 x i64` straight forwardly calculated as:

```rust
(i32, i32)
(i32, i64)
(i64, i32)
(i64, i64)
```

This is useful because all possible variants can directly be computed as a product:
`f64 x i32 x f32 x String`. Types which behave in this fashion have the properties
of a product type and so are categorized as such.

#### Sum types

Rust's version of sum types is the enum. An enum is defined as a type which can be
one of the variants only. For example:

```rust
// Define `Number` to be either a small number `Small` or a big number `Big`.
enum Number {
	Small(i32),
	Big(i64),
}
```

In contrast to a product type where given two variants `i32 x i64`, there will be two
data values in the structure of any order, a sum type can only ever have one of the
variants. This means that for a type to be of `i32 + i64`, it will be either of the
two.

They are called tagged union because each variant is tagged to distinguish them:

```rust
// This enum `Color` has variants which hold no data. They are tagged to distinguish
// between each other.
enum Color {
	Red,
	Green,
	Blue,
}
```

This allows using names to distinguish something rather than a number. A name is much
more descriptive. For example, errors might better be represented with names than
strictly numbers.
