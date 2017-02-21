# Expressions

An expression may have two roles: it always produces a *value*, and it may have
*effects* (otherwise known as "side effects"). An expression *evaluates to* a
value, and has effects during *evaluation*. Many expressions contain
sub-expressions (operands). The meaning of each kind of expression dictates
several things:

* Whether or not to evaluate the sub-expressions when evaluating the expression
* The order in which to evaluate the sub-expressions
* How to combine the sub-expressions' values to obtain the value of the expression

In this way, the structure of expressions dictates the structure of execution.
Blocks are just another kind of expression, so blocks, statements, expressions,
and blocks again can recursively nest inside each other to an arbitrary depth.

### Lvalues, rvalues and temporaries

Expressions are divided into two main categories: _lvalues_ and _rvalues_.
Likewise within each expression, sub-expressions may occur in _lvalue context_
or _rvalue context_. The evaluation of an expression depends both on its own
category and the context it occurs within.

An lvalue is an expression that represents a memory location. These expressions
are [paths](#path-expressions) (which refer to local variables, function and
method arguments, or static variables), dereferences (`*expr`), [indexing
expressions](#index-expressions) (`expr[expr]`), and [field
references](#field-expressions) (`expr.f`). All other expressions are rvalues.

The left operand of an [assignment](#assignment-expressions) or
[compound-assignment](#compound-assignment-expressions) expression is
an lvalue context, as is the single operand of a unary
[borrow](#unary-operator-expressions). The discriminant or subject of
a [match expression](#match-expressions) may be an lvalue context, if
ref bindings are made, but is otherwise an rvalue context. All other
expression contexts are rvalue contexts.

When an lvalue is evaluated in an _lvalue context_, it denotes a memory
location; when evaluated in an _rvalue context_, it denotes the value held _in_
that memory location.

#### Temporary lifetimes

When an rvalue is used in an lvalue context, a temporary un-named
lvalue is created and used instead. The lifetime of temporary values
is typically the innermost enclosing statement; the tail expression of
a block is considered part of the statement that encloses the block.

When a temporary rvalue is being created that is assigned into a `let`
declaration, however, the temporary is created with the lifetime of
the enclosing block instead, as using the enclosing statement (the
`let` declaration) would be a guaranteed error (since a pointer to the
temporary would be stored into a variable, but the temporary would be
freed before the variable could be used). The compiler uses simple
syntactic rules to decide which values are being assigned into a `let`
binding, and therefore deserve a longer temporary lifetime.

Here are some examples:

- `let x = foo(&temp())`. The expression `temp()` is an rvalue. As it
  is being borrowed, a temporary is created which will be freed after
  the innermost enclosing statement (the `let` declaration, in this case).
- `let x = temp().foo()`. This is the same as the previous example,
  except that the value of `temp()` is being borrowed via autoref on a
  method-call. Here we are assuming that `foo()` is an `&self` method
  defined in some trait, say `Foo`. In other words, the expression
  `temp().foo()` is equivalent to `Foo::foo(&temp())`.
- `let x = &temp()`. Here, the same temporary is being assigned into
  `x`, rather than being passed as a parameter, and hence the
  temporary's lifetime is considered to be the enclosing block.
- `let x = SomeStruct { foo: &temp() }`. As in the previous case, the
  temporary is assigned into a struct which is then assigned into a
  binding, and hence it is given the lifetime of the enclosing block.
- `let x = [ &temp() ]`. As in the previous case, the
  temporary is assigned into an array which is then assigned into a
  binding, and hence it is given the lifetime of the enclosing block.
- `let ref x = temp()`. In this case, the temporary is created using a ref binding,
  but the result is the same: the lifetime is extended to the enclosing block.

### Moved and copied types

When a [local variable](variables.html) is used as an
[rvalue](expressions.html#lvalues-rvalues-and-temporaries), the variable will
be copied if its type implements `Copy`. All others are moved.

## Literal expressions

A _literal expression_ consists of one of the [literal](tokens.html#literals) forms
described earlier. It directly describes a number, character, string, boolean
value, or the unit value.

```text
();        // unit type
"hello";   // string type
'5';       // character type
5;         // integer type
```

## Path expressions

A [path](paths.html) used as an expression context denotes either a local
variable or an item. Path expressions are
[lvalues](expressions.html#lvalues-rvalues-and-temporaries).

## Tuple expressions

Tuples are written by enclosing zero or more comma-separated expressions in
parentheses. They are used to create [tuple-typed](types.html#tuple-types)
values.

```{.tuple}
(0.0, 4.5);
("a", 4usize, true);
```

You can disambiguate a single-element tuple from a value in parentheses with a
comma:

```
(0,); // single-element tuple
(0); // zero in parentheses
```

## Struct expressions

There are several forms of struct expressions. A _struct expression_
consists of the [path](paths.html) of a [struct item](items.html#structs), followed
by a brace-enclosed list of zero or more comma-separated name-value pairs,
providing the field values of a new instance of the struct. A field name can be
any identifier, and is separated from its value expression by a colon.  The
location denoted by a struct field is mutable if and only if the enclosing
struct is mutable.

A _tuple struct expression_ consists of the [path](paths.html) of a [struct
item](items.html#structs), followed by a parenthesized list of one or more
comma-separated expressions (in other words, the path of a struct item followed
by a tuple expression). The struct item must be a tuple struct item.

A _unit-like struct expression_ consists only of the [path](paths.html) of a
[struct item](items.html#structs).

The following are examples of struct expressions:

```
# struct Point { x: f64, y: f64 }
# struct NothingInMe { }
# struct TuplePoint(f64, f64);
# mod game { pub struct User<'a> { pub name: &'a str, pub age: u32, pub score: usize } }
# struct Cookie; fn some_fn<T>(t: T) {}
Point {x: 10.0, y: 20.0};
NothingInMe {};
TuplePoint(10.0, 20.0);
let u = game::User {name: "Joe", age: 35, score: 100_000};
some_fn::<Cookie>(Cookie);
```

A struct expression forms a new value of the named struct type. Note
that for a given *unit-like* struct type, this will always be the same
value.

A struct expression can terminate with the syntax `..` followed by an
expression to denote a functional update. The expression following `..` (the
base) must have the same struct type as the new struct type being formed.
The entire expression denotes the result of constructing a new struct (with
the same type as the base expression) with the given values for the fields that
were explicitly specified and the values in the base expression for all other
fields.

```
# struct Point3d { x: i32, y: i32, z: i32 }
let base = Point3d {x: 1, y: 2, z: 3};
Point3d {y: 0, z: 10, .. base};
```

#### Struct field init shorthand

When initializing a data structure (struct, enum, union) with named fields,
it is allowed to write `fieldname` as a shorthand for `fieldname: fieldname`.
This allows a compact syntax with less duplication.

Example:

```
# struct Point3d { x: i32, y: i32, z: i32 }
# let x = 0;
# let y_value = 0;
# let z = 0;
Point3d { x: x, y: y_value, z: z };
Point3d { x, y: y_value, z };
```

## Block expressions

A _block expression_ is similar to a module in terms of the declarations that
are possible. Each block conceptually introduces a new namespace scope. Use
items can bring new names into scopes and declared items are in scope for only
the block itself.

A block will execute each statement sequentially, and then execute the
expression (if given). If the block ends in a statement, its value is `()`:

```
let x: () = { println!("Hello."); };
```

If it ends in an expression, its value and type are that of the expression:

```
let x: i32 = { println!("Hello."); 5 };

assert_eq!(5, x);
```

## Method-call expressions

A _method call_ consists of an expression followed by a single dot, an
identifier, and a parenthesized expression-list. Method calls are resolved to
methods on specific traits, either statically dispatching to a method if the
exact `self`-type of the left-hand-side is known, or dynamically dispatching if
the left-hand-side expression is an indirect [trait
object](types.html#trait-objects).

## Field expressions

A _field expression_ consists of an expression followed by a single dot and an
identifier, when not immediately followed by a parenthesized expression-list
(the latter is a [method call expression](#method-call-expressions)). A field
expression denotes a field of a [struct](types.html#struct-types).

```{.ignore .field}
mystruct.myfield;
foo().x;
(Struct {a: 10, b: 20}).a;
```

A field access is an [lvalue](expressions.html#lvalues-rvalues-and-temporaries)
referring to the value of that field. When the type providing the field
inherits mutability, it can be [assigned](#assignment-expressions) to.

Also, if the type of the expression to the left of the dot is a
pointer, it is automatically dereferenced as many times as necessary
to make the field access possible. In cases of ambiguity, we prefer
fewer autoderefs to more.

## Array expressions

An [array](types.html#array-and-slice-types) _expression_ is written by
enclosing zero or more comma-separated expressions of uniform type in square
brackets.

In the `[expr ';' expr]` form, the expression after the `';'` must be a
constant expression that can be evaluated at compile time, such as a
[literal](tokens.html#literals) or a [static item](items.html#static-items).

```
[1, 2, 3, 4];
["a", "b", "c", "d"];
[0; 128];              // array with 128 zeros
[0u8, 0u8, 0u8, 0u8];
```

## Index expressions

[Array](types.html#array-and-slice-types)-typed expressions can be indexed by
writing a square-bracket-enclosed expression (the index) after them. When the
array is mutable, the resulting
[lvalue](expressions.html#lvalues-rvalues-and-temporaries) can be assigned to.

Indices are zero-based, and may be of any integral type. Vector access is
bounds-checked at compile-time for constant arrays being accessed with a
constant index value.  Otherwise a check will be performed at run-time that
will put the thread in a _panicked state_ if it fails.

```{should-fail}
([1, 2, 3, 4])[0];

let x = (["a", "b"])[10]; // compiler error: const index-expr is out of bounds

let n = 10;
let y = (["a", "b"])[n]; // panics

let arr = ["a", "b"];
arr[10]; // panics
```

Also, if the type of the expression to the left of the brackets is a
pointer, it is automatically dereferenced as many times as necessary
to make the indexing possible. In cases of ambiguity, we prefer fewer
autoderefs to more.

## Range expressions

The `..` operator will construct an object of one of the `std::ops::Range` variants.

```
1..2;   // std::ops::Range
3..;    // std::ops::RangeFrom
..4;    // std::ops::RangeTo
..;     // std::ops::RangeFull
```

The following expressions are equivalent.

```
let x = std::ops::Range {start: 0, end: 10};
let y = 0..10;

assert_eq!(x, y);
```

Similarly, the `...` operator will construct an object of one of the
`std::ops::RangeInclusive` variants.

```
# #![feature(inclusive_range_syntax)]
1...2;   // std::ops::RangeInclusive
...4;    // std::ops::RangeToInclusive
```

The following expressions are equivalent.

```
# #![feature(inclusive_range_syntax, inclusive_range)]
let x = std::ops::RangeInclusive::NonEmpty {start: 0, end: 10};
let y = 0...10;

assert_eq!(x, y);
```

## Unary operator expressions

Rust defines the following unary operators. With the exception of `?`, they are
all written as prefix operators, before the expression they apply to.

* `-`
  : Negation. Signed integer types and floating-point types support negation. It
    is an error to apply negation to unsigned types; for example, the compiler
    rejects `-1u32`.
* `*`
  : Dereference. When applied to a [pointer](types.html#pointer-types) it
    denotes the pointed-to location. For pointers to mutable locations, the
    resulting [lvalue](expressions.html#lvalues-rvalues-and-temporaries) can be
    assigned to.  On non-pointer types, it calls the `deref` method of the
    `std::ops::Deref` trait, or the `deref_mut` method of the
    `std::ops::DerefMut` trait (if implemented by the type and required for an
    outer expression that will or could mutate the dereference), and produces
    the result of dereferencing the `&` or `&mut` borrowed pointer returned
    from the overload method.
* `!`
  : Logical negation. On the boolean type, this flips between `true` and
    `false`. On integer types, this inverts the individual bits in the
    two's complement representation of the value.
* `&` and `&mut`
  : Borrowing. When applied to an lvalue, these operators produce a
    reference (pointer) to the lvalue. The lvalue is also placed into
    a borrowed state for the duration of the reference. For a shared
    borrow (`&`), this implies that the lvalue may not be mutated, but
    it may be read or shared again. For a mutable borrow (`&mut`), the
    lvalue may not be accessed in any way until the borrow expires.
    If the `&` or `&mut` operators are applied to an rvalue, a
    temporary value is created; the lifetime of this temporary value
    is defined by [syntactic rules](#temporary-lifetimes).
* `?`
  : Propagating errors if applied to `Err(_)` and unwrapping if
    applied to `Ok(_)`. Only works on the `Result<T, E>` type,
    and written in postfix notation.

## Binary operator expressions

Binary operators expressions are given in terms of [operator
precedence](#operator-precedence).

### Arithmetic operators

Binary arithmetic expressions are syntactic sugar for calls to built-in traits,
defined in the `std::ops` module of the `std` library. This means that
arithmetic operators can be overridden for user-defined types. The default
meaning of the operators on standard types is given here.

* `+`
  : Addition and array/string concatenation.
    Calls the `add` method on the `std::ops::Add` trait.
* `-`
  : Subtraction.
    Calls the `sub` method on the `std::ops::Sub` trait.
* `*`
  : Multiplication.
    Calls the `mul` method on the `std::ops::Mul` trait.
* `/`
  : Quotient.
    Calls the `div` method on the `std::ops::Div` trait.
* `%`
  : Remainder.
    Calls the `rem` method on the `std::ops::Rem` trait.

### Bitwise operators

Like the [arithmetic operators](#arithmetic-operators), bitwise operators are
syntactic sugar for calls to methods of built-in traits. This means that
bitwise operators can be overridden for user-defined types. The default
meaning of the operators on standard types is given here. Bitwise `&`, `|` and
`^` applied to boolean arguments are equivalent to logical `&&`, `||` and `!=`
evaluated in non-lazy fashion.

* `&`
  : Bitwise AND.
    Calls the `bitand` method of the `std::ops::BitAnd` trait.
* `|`
  : Bitwise inclusive OR.
    Calls the `bitor` method of the `std::ops::BitOr` trait.
* `^`
  : Bitwise exclusive OR.
    Calls the `bitxor` method of the `std::ops::BitXor` trait.
* `<<`
  : Left shift.
    Calls the `shl` method of the `std::ops::Shl` trait.
* `>>`
  : Right shift (arithmetic).
    Calls the `shr` method of the `std::ops::Shr` trait.

### Lazy boolean operators

The operators `||` and `&&` may be applied to operands of boolean type. The
`||` operator denotes logical 'or', and the `&&` operator denotes logical
'and'. They differ from `|` and `&` in that the right-hand operand is only
evaluated when the left-hand operand does not already determine the result of
the expression. That is, `||` only evaluates its right-hand operand when the
left-hand operand evaluates to `false`, and `&&` only when it evaluates to
`true`.

### Comparison operators

Comparison operators are, like the [arithmetic
operators](#arithmetic-operators), and [bitwise operators](#bitwise-operators),
syntactic sugar for calls to built-in traits. This means that comparison
operators can be overridden for user-defined types. The default meaning of the
operators on standard types is given here.

* `==`
  : Equal to.
    Calls the `eq` method on the `std::cmp::PartialEq` trait.
* `!=`
  : Unequal to.
    Calls the `ne` method on the `std::cmp::PartialEq` trait.
* `<`
  : Less than.
    Calls the `lt` method on the `std::cmp::PartialOrd` trait.
* `>`
  : Greater than.
    Calls the `gt` method on the `std::cmp::PartialOrd` trait.
* `<=`
  : Less than or equal.
    Calls the `le` method on the `std::cmp::PartialOrd` trait.
* `>=`
  : Greater than or equal.
    Calls the `ge` method on the `std::cmp::PartialOrd` trait.

### Type cast expressions

A type cast expression is denoted with the binary operator `as`.

Executing an `as` expression casts the value on the left-hand side to the type
on the right-hand side.

An example of an `as` expression:

```
# fn sum(values: &[f64]) -> f64 { 0.0 }
# fn len(values: &[f64]) -> i32 { 0 }

fn average(values: &[f64]) -> f64 {
    let sum: f64 = sum(values);
    let size: f64 = len(values) as f64;
    sum / size
}
```

Some of the conversions which can be done through the `as` operator
can also be done implicitly at various points in the program, such as
argument passing and assignment to a `let` binding with an explicit
type. Implicit conversions are limited to "harmless" conversions that
do not lose information and which have minimal or no risk of
surprising side-effects on the dynamic execution semantics.

### Assignment expressions

An _assignment expression_ consists of an
[lvalue](expressions.html#lvalues-rvalues-and-temporaries) expression followed
by an equals sign (`=`) and an
[rvalue](expressions.html#lvalues-rvalues-and-temporaries) expression.

Evaluating an assignment expression [either copies or
moves](#moved-and-copied-types) its right-hand operand to its left-hand
operand.

```
# let mut x = 0;
# let y = 0;
x = y;
```

### Compound assignment expressions

The `+`, `-`, `*`, `/`, `%`, `&`, `|`, `^`, `<<`, and `>>` operators may be
composed with the `=` operator. The expression `lval OP= val` is equivalent to
`lval = lval OP val`. For example, `x = x + 1` may be written as `x += 1`.

Any such expression always has the [`unit`](types.html#tuple-types) type.

### Operator precedence

The precedence of Rust binary operators is ordered as follows, going from
strong to weak:

```{.text .precedence}
as :
* / %
+ -
<< >>
&
^
|
== != < > <= >=
&&
||
.. ...
<-
=
```

Operators at the same precedence level are evaluated left-to-right. [Unary
operators](#unary-operator-expressions) have the same precedence level and are
stronger than any of the binary operators.

## Grouped expressions

An expression enclosed in parentheses evaluates to the result of the enclosed
expression. Parentheses can be used to explicitly specify evaluation order
within an expression.

An example of a parenthesized expression:

```
let x: i32 = (2 + 3) * 4;
```


## Call expressions

A _call expression_ invokes a function, providing zero or more input variables
and an optional location to move the function's output into. If the function
eventually returns, then the expression completes.

Some examples of call expressions:

```
# fn add(x: i32, y: i32) -> i32 { 0 }

let x: i32 = add(1i32, 2i32);
let pi: Result<f32, _> = "3.14".parse();
```

## Lambda expressions

A _lambda expression_ (sometimes called an "anonymous function expression")
defines a function and denotes it as a value, in a single expression. A lambda
expression is a pipe-symbol-delimited (`|`) list of identifiers followed by an
expression.

A lambda expression denotes a function that maps a list of parameters
(`ident_list`) onto the expression that follows the `ident_list`. The
identifiers in the `ident_list` are the parameters to the function. These
parameters' types need not be specified, as the compiler infers them from
context.

Lambda expressions are most useful when passing functions as arguments to other
functions, as an abbreviation for defining and capturing a separate function.

Significantly, lambda expressions _capture their environment_, which regular
[function definitions](items.html#functions) do not. The exact type of capture
depends on the [function type](types.html#function-types) inferred for the
lambda expression. In the simplest and least-expensive form (analogous to a
```|| { }``` expression), the lambda expression captures its environment by
reference, effectively borrowing pointers to all outer variables mentioned
inside the function.  Alternately, the compiler may infer that a lambda
expression should copy or move values (depending on their type) from the
environment into the lambda expression's captured environment. A lambda can be
forced to capture its environment by moving values by prefixing it with the
`move` keyword.

In this example, we define a function `ten_times` that takes a higher-order
function argument, and we then call it with a lambda expression as an argument,
followed by a lambda expression that moves values from its environment.

```
fn ten_times<F>(f: F) where F: Fn(i32) {
    for index in 0..10 {
        f(index);
    }
}

ten_times(|j| println!("hello, {}", j));

let word = "konnichiwa".to_owned();
ten_times(move |j| println!("{}, {}", word, j));
```

## Infinite loops

A `loop` expression denotes an infinite loop.

A `loop` expression may optionally have a _label_. The label is written as
a lifetime preceding the loop expression, as in `'foo: loop{ }`. If a
label is present, then labeled `break` and `continue` expressions nested
within this loop may exit out of this loop or return control to its head.
See [break expressions](#break-expressions) and [continue
expressions](#continue-expressions).

## `break` expressions

A `break` expression has an optional _label_. If the label is absent, then
executing a `break` expression immediately terminates the innermost loop
enclosing it. It is only permitted in the body of a loop. If the label is
present, then `break 'foo` terminates the loop with label `'foo`, which need not
be the innermost label enclosing the `break` expression, but must enclose it.

## `continue` expressions

A `continue` expression has an optional _label_. If the label is absent, then
executing a `continue` expression immediately terminates the current iteration
of the innermost loop enclosing it, returning control to the loop *head*. In
the case of a `while` loop, the head is the conditional expression controlling
the loop. In the case of a `for` loop, the head is the call-expression
controlling the loop. If the label is present, then `continue 'foo` returns
control to the head of the loop with label `'foo`, which need not be the
innermost label enclosing the `continue` expression, but must enclose it.

A `continue` expression is only permitted in the body of a loop.

## `while` loops

A `while` loop begins by evaluating the boolean loop conditional expression.
If the loop conditional expression evaluates to `true`, the loop body block
executes and control returns to the loop conditional expression. If the loop
conditional expression evaluates to `false`, the `while` expression completes.

An example:

```
let mut i = 0;

while i < 10 {
    println!("hello");
    i = i + 1;
}
```

Like `loop` expressions, `while` loops can be controlled with `break` or
`continue`, and may optionally have a _label_. See [infinite
loops](#infinite-loops), [break expressions](#break-expressions), and
[continue expressions](#continue-expressions) for more information.

## `for` expressions

A `for` expression is a syntactic construct for looping over elements provided
by an implementation of `std::iter::IntoIterator`.

An example of a `for` loop over the contents of an array:

```
# type Foo = i32;
# fn bar(f: &Foo) { }
# let a = 0;
# let b = 0;
# let c = 0;

let v: &[Foo] = &[a, b, c];

for e in v {
    bar(e);
}
```

An example of a for loop over a series of integers:

```
# fn bar(b:usize) { }
for i in 0..256 {
    bar(i);
}
```

Like `loop` expressions, `for` loops can be controlled with `break` or
`continue`, and may optionally have a _label_. See [infinite
loops](#infinite-loops), [break expressions](#break-expressions), and
[continue expressions](#continue-expressions) for more information.

## `if` expressions

An `if` expression is a conditional branch in program control. The form of an
`if` expression is a condition expression, followed by a consequent block, any
number of `else if` conditions and blocks, and an optional trailing `else`
block. The condition expressions must have type `bool`. If a condition
expression evaluates to `true`, the consequent block is executed and any
subsequent `else if` or `else` block is skipped. If a condition expression
evaluates to `false`, the consequent block is skipped and any subsequent `else
if` condition is evaluated. If all `if` and `else if` conditions evaluate to
`false` then any `else` block is executed.

## `match` expressions

A `match` expression branches on a *pattern*. The exact form of matching that
occurs depends on the pattern. Patterns consist of some combination of
literals, destructured arrays or enum constructors, structs and tuples,
variable binding specifications, wildcards (`..`), and placeholders (`_`). A
`match` expression has a *head expression*, which is the value to compare to
the patterns. The type of the patterns must equal the type of the head
expression.

In a pattern whose head expression has an `enum` type, a placeholder (`_`)
stands for a *single* data field, whereas a wildcard `..` stands for *all* the
fields of a particular variant.

A `match` behaves differently depending on whether or not the head expression
is an [lvalue or an rvalue](expressions.html#lvalues-rvalues-and-temporaries).
If the head expression is an rvalue, it is first evaluated into a temporary
location, and the resulting value is sequentially compared to the patterns in
the arms until a match is found. The first arm with a matching pattern is
chosen as the branch target of the `match`, any variables bound by the pattern
are assigned to local variables in the arm's block, and control enters the
block.

When the head expression is an lvalue, the match does not allocate a temporary
location (however, a by-value binding may copy or move from the lvalue). When
possible, it is preferable to match on lvalues, as the lifetime of these
matches inherits the lifetime of the lvalue, rather than being restricted to
the inside of the match.

An example of a `match` expression:

```
let x = 1;

match x {
    1 => println!("one"),
    2 => println!("two"),
    3 => println!("three"),
    4 => println!("four"),
    5 => println!("five"),
    _ => println!("something else"),
}
```

Patterns that bind variables default to binding to a copy or move of the
matched value (depending on the matched value's type). This can be changed to
bind to a reference by using the `ref` keyword, or to a mutable reference using
`ref mut`.

Subpatterns can also be bound to variables by the use of the syntax `variable @
subpattern`. For example:

```
let x = 1;

match x {
    e @ 1 ... 5 => println!("got a range element {}", e),
    _ => println!("anything"),
}
```

Patterns can also dereference pointers by using the `&`, `&mut` and `box`
symbols, as appropriate. For example, these two matches on `x: &i32` are
equivalent:

```
# let x = &3;
let y = match *x { 0 => "zero", _ => "some" };
let z = match x { &0 => "zero", _ => "some" };

assert_eq!(y, z);
```

Multiple match patterns may be joined with the `|` operator. A range of values
may be specified with `...`. For example:

```
# let x = 2;

let message = match x {
    0 | 1  => "not many",
    2 ... 9 => "a few",
    _      => "lots"
};
```

Range patterns only work on scalar types (like integers and characters; not
like arrays and structs, which have sub-components). A range pattern may not
be a sub-range of another range pattern inside the same `match`.

Finally, match patterns can accept *pattern guards* to further refine the
criteria for matching a case. Pattern guards appear after the pattern and
consist of a bool-typed expression following the `if` keyword. A pattern guard
may refer to the variables bound within the pattern they follow.

```
# let maybe_digit = Some(0);
# fn process_digit(i: i32) { }
# fn process_other(i: i32) { }

let message = match maybe_digit {
    Some(x) if x < 10 => process_digit(x),
    Some(x) => process_other(x),
    None => panic!(),
};
```

## `if let` expressions

An `if let` expression is semantically identical to an `if` expression but in
place of a condition expression it expects a `let` statement with a refutable
pattern. If the value of the expression on the right hand side of the `let`
statement matches the pattern, the corresponding block will execute, otherwise
flow proceeds to the first `else` block that follows.

```
let dish = ("Ham", "Eggs");

// this body will be skipped because the pattern is refuted
if let ("Bacon", b) = dish {
    println!("Bacon is served with {}", b);
}

// this body will execute
if let ("Ham", b) = dish {
    println!("Ham is served with {}", b);
}
```

## `while let` loops

A `while let` loop is semantically identical to a `while` loop but in place of
a condition expression it expects `let` statement with a refutable pattern. If
the value of the expression on the right hand side of the `let` statement
matches the pattern, the loop body block executes and control returns to the
pattern matching statement. Otherwise, the while expression completes.

## `return` expressions

Return expressions are denoted with the keyword `return`. Evaluating a `return`
expression moves its argument into the designated output location for the
current function call, destroys the current function activation frame, and
transfers control to the caller frame.

An example of a `return` expression:

```
fn max(a: i32, b: i32) -> i32 {
    if a > b {
        return a;
    }
    return b;
}
```
