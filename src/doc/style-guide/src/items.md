## Items

Items consist of the set of things permitted at the top level of a module.
However, Rust also allows some items to appear within some other types of
items, such as within a function. The same formatting conventions apply whether
an item appears at module level or within another item.

`extern crate` statements must be first in a file. They must be ordered
alphabetically.

`use` statements, and module *declarations* (`mod foo;`, not `mod { ... }`)
must come before other items. We recommend that imports come before module
declarations; if imports and modules are separated, then they should be ordered
alphabetically. When sorting, `self` and `super` must come before any other
names. Module declarations should not be moved if they are annotated with
`#[macro_use]`, since that may be semantics changing.

Tools should make the above ordering optional.


### Function definitions

In Rust, people often find functions by searching for `fn [function-name]`, so
the formatting of function definitions shold enable this.

The proper ordering and spacing is:

```rust
[pub] [unsafe] [extern ["ABI"]] fn foo(arg1: i32, arg2: i32) -> i32 {
    ...
}
```

Avoid comments within the signature itself.

If the function signature does not fit on one line, then break after the opening
parenthesis and before the closing parenthesis and put each argument on its own
block-indented line. For example,

```rust
fn foo(
    arg1: i32,
    arg2: i32,
) -> i32 {
    ...
}
```

Note the trailing comma on the last argument.


### Tuples and tuple structs

Write the type list as you would a parameter list to a function.

Build a tuple or tuple struct as you would call a function.

#### Single-line

```rust
struct Bar(Type1, Type2);

let x = Bar(11, 22);
let y = (11, 22, 33);
```

### Enums

In the declaration, put each variant on its own line, block indented.

Format each variant accordingly as either a struct (but without the `struct`
keyword), a tuple struct, or an identifier (which doesn't require special
formatting):

```rust
enum FooBar {
    First(u32),
    Second,
    Error {
        err: Box<Error>,
        line: u32,
    },
}
```

If a struct variant is [*small*](index.html#small-items), it may be formatted on
one line. In this case, do not use a trailing comma for the field list, but do
put spaces around each brace:

```rust
enum FooBar {
    Error { err: Box<Error>, line: u32 },
}
```

In an enum with multiple struct variants, if any struct variant is written on
multiple lines, then the multi-line formatting should be used for all struct
variants. However, such a situation might be an indication that you should
factor out the fields of the variant into their own struct.


### Structs and Unions

Struct names follow on the same line as the `struct` keyword, with the opening
brace on the same line when it fits within the right margin. All struct fields
are indented once and end with a trailing comma. The closing brace is not
indented and appears on its own line.

```rust
struct Foo {
    a: A,
    b: B,
}
```

If and only if the type of a field does not fit within the right margin, it is
pulled down to its own line and indented again.

```rust
struct Foo {
    a: A,
    long_name:
        LongType,
}
```

Prefer using a unit struct (e.g., `struct Foo;`) to an empty struct (e.g.,
`struct Foo();` or `struct Foo {}`, these only exist to simplify code
generation), but if you must use an empty struct, keep it on one line with no
space between the braces: `struct Foo;` or `struct Foo {}`.

The same guidelines are used for untagged union declarations.

```rust
union Foo {
    a: A,
    b: B,
    long_name:
        LongType,
}
```


### Tuple structs

Put the whole struct on one line if possible. Types in the parentheses should be
separated by a comma and space with no trailing comma. No spaces around the
parentheses or semicolon:

```rust
pub struct Foo(String, u8);
```

Prefer unit structs to empty tuple structs (these only exist to simplify code
generation), e.g., `struct Foo;` rather than `struct Foo();`.

For more than a few fields, prefer a proper struct with named fields. Given
this, a tuple struct should always fit on one line. If it does not, block format
the fields with a field on each line and a trailing comma:

```rust
pub struct Foo(
    String,
    u8,
);
```


### Traits

Trait items should be block-indented. If there are no items, the trait may be
formatted on a single line. Otherwise there should be line-breaks after the
opening brace and before the closing brace:

```rust
trait Foo {}

pub trait Bar {
    ...
}
```

If the trait has bounds, there should be a space after the colon but not before
and before and after each `+`, e.g.,

```rust
trait Foo: Debug + Bar {}
```

Prefer not to line-break in the bounds if possible (consider using a `where`
clause). Prefer to break between bounds than to break any individual bound. If
you must break the bounds, put each bound (including the first) on its own
block-indented line, break before the `+` and put the opening brace on its own
line:

```rust
pub trait IndexRanges:
    Index<Range<usize>, Output=Self>
    + Index<RangeTo<usize>, Output=Self>
    + Index<RangeFrom<usize>, Output=Self>
    + Index<RangeFull, Output=Self>
{
    ...
}
```


### Impls

Impl items should be block indented. If there are no items, the impl may be
formatted on a single line. Otherwise there should be line-breaks after the
opening brace and before the closing brace:

```rust
impl Foo {}

impl Bar for Foo {
    ...
}
```

Avoid line-breaking in the signature if possible. If a line break is required in
a non-inherent impl, break immediately before `for`, block indent the concrete type
and put the opening brace on its own line:

```rust
impl Bar
    for Foo
{
    ...
}
```


### Extern crate

`extern crate foo;`

Use spaces around keywords, no spaces around the semicolon.


### Modules

```rust
mod foo {
}
```

```rust
mod foo;
```

Use spaces around keywords and before the opening brace, no spaces around the
semicolon.

### macro\_rules!

Use `{}` for the full definition of the macro.

```rust
macro_rules! foo {
}
```


### Generics

Prefer to put a generics clause on one line. Break other parts of an item
declaration rather than line-breaking a generics clause. If a generics clause is
large enough to require line-breaking, you should prefer to use a `where` clause
instead.

Do not put spaces before or after `<` nor before `>`. Only put a space after `>`
if it is followed by a word or opening brace, not an opening parenthesis. There
should be a space after each comma and no trailing comma.

```rust
fn foo<T: Display, U: Debug>(x: Vec<T>, y: Vec<U>) ...

impl<T: Display, U: Debug> SomeType<T, U> { ...
```

If the generics clause must be formatted across multiple lines, each parameter
should have its own block-indented line, there should be newlines after the
opening bracket and before the closing bracket, and the should be a trailing
comma.

```rust
fn foo<
    T: Display,
    U: Debug,
>(x: Vec<T>, y: Vec<U>) ...
```

If an associated type is bound in a generic type, then there should be spaces on
either side of the `=`:

```rust
<T: Example<Item = u32>>
```

Prefer to use single-letter names for generic parameters.


### `where` clauses

These rules apply for `where` clauses on any item.

A `where` clause may immediately follow a closing bracket of any kind.
Otherwise, it must start a new line, with no indent. Each component of a `where`
clause must be on its own line and be block indented. There should be a trailing
comma, unless the clause is terminated with a semicolon. If the `where` clause
is followed by a block (or assignment), the block should be started on a new
line. Examples:

```rust
fn function<T, U>(args)
where
    T: Bound,
    U: AnotherBound,
{
    body
}

fn foo<T>(
    args
) -> ReturnType
where
    T: Bound,
{
    body
}

fn foo<T, U>(
    args,
) where
    T: Bound,
    U: AnotherBound,
{
    body
}

fn foo<T, U>(
    args
) -> ReturnType
where
    T: Bound,
    U: AnotherBound;  // Note, no trailing comma.

// Note that where clauses on `type` aliases are not enforced and should not
// be used.
type Foo<T>
where
    T: Bound
= Bar<T>;
```

If a `where` clause is very short, we recommend using an inline bound on the
type parameter.


If a component of a `where` clause is long, it may be broken before `+` and
further block indented. Each bound should go on its own line. E.g.,

```rust
impl<T: ?Sized, Idx> IndexRanges<Idx> for T
where
    T: Index<Range<Idx>, Output = Self::Output>
        + Index<RangeTo<Idx>, Output = Self::Output>
        + Index<RangeFrom<Idx>, Output = Self::Output>
        + Index<RangeInclusive<Idx>, Output = Self::Output>
        + Index<RangeToInclusive<Idx>, Output = Self::Output> + Index<RangeFull>
```

#### Option - `where_single_line`

`where_single_line` is `false` by default. If `true`, then a where clause with
exactly one component may be formatted on a single line if the rest of the
item's signature is also kept on one line. In this case, there is no need for a
trailing comma and if followed by a block, no need for a newline before the
block. E.g.,

```rust
// May be single-lined.
fn foo<T>(args) -> ReturnType
where T: Bound {
    body
}

// Must be multi-lined.
fn foo<T>(
    args
) -> ReturnType
where
    T: Bound,
{
    body
}
```


### Type aliases

Type aliases should generally be kept on one line. If necessary to break the
line, do so after the `=`; the right-hand-side should be block indented:

```rust
pub type Foo = Bar<T>;

// If multi-line is required
type VeryLongType<T, U: SomeBound> =
    AnEvenLongerType<T, U, Foo<T>>;
```

Where possible avoid `where` clauses and keep type constraints inline. Where
that is not possible split the line before and after the `where` clause (and
split the `where` clause as normal), e.g.,

```rust
type VeryLongType<T, U>
where
    T: U::AnAssociatedType,
    U: SomeBound,
= AnEvenLongerType<T, U, Foo<T>>;
```


### Associated types

Associated types should follow the guidelines above for type aliases. Where an
associated type has a bound, there should be a space after the colon but not
before:

```rust
pub type Foo: Bar;
```


### extern items

When writing extern items (such as `extern "C" fn`), always be explicit about
the ABI. For example, write `extern "C" fn foo ...`, not `extern fn foo ...`, or
`extern "C" { ... }`.


### Imports (`use` statements)

If an import can be formatted on one line, do so. There should be no spaces
around braces.

```rust
use a::b::c;
use a::b::d::*;
use a::b::{foo, bar, baz};
```


#### Large list imports

Prefer to use multiple imports rather than a multi-line import. However, tools
should not split imports by default (they may offer this as an option).

If an import does require multiple lines (either because a list of single names
does not fit within the max width, or because of the rules for nested imports
below), then break after the opening brace and before the closing brace, use a
trailing comma, and block indent the names.


```rust
// Prefer
foo::{long, list, of, imports};
foo::{more, imports};

// If necessary
foo::{
    long, list, of, imports, more,
    imports,  // Note trailing comma
};
```


#### Ordering of imports

A *group* of imports is a set of imports on the same or sequential lines. One or
more blank lines or other items (e.g., a function) separate groups of imports.

Within a group of imports, imports must be sorted ASCIIbetically (uppercase
before lowercase). Groups of imports must not be merged or re-ordered.


E.g., input:

```rust
use d;
use c;

use b;
use a;
```

output:

```rust
use c;
use d;

use a;
use b;
```

Because of `macro_use`, attributes must also start a new group and prevent
re-ordering.

#### Ordering list import

Names in a list import must be sorted ASCIIbetically, but with `self` and
`super` first, and groups and glob imports last. This applies recursively. For
example, `a::*` comes before `b::a` but `a::b` comes before `a::*`. E.g.,
`use foo::bar::{a, b::c, b::d, b::d::{x, y, z}, b::{self, r, s}};`.


#### Normalisation

Tools must make the following normalisations:

* `use a::self;` -> `use a;`
* `use a::{};` -> (nothing)
* `use a::{b};` -> `use a::b;`

And must apply these recursively.

Tools must not otherwise merge or un-merge import lists or adjust glob imports
(without an explicit option).


#### Nested imports

If there are any nested imports in a list import, then use the multi-line form,
even if the import fits on one line. Each nested import must be on its own line,
but non-nested imports must be grouped on as few lines as possible.

For example,

```rust
use a::b::{
    x, y, z,
    u::{...},
    w::{...},
};
```


#### Merging/un-merging imports

An example:

```rust
// Un-merged
use a::b;
use a::c::d;

// Merged
use a::{b, c::d};
```

Tools must not merge or un-merge imports by default. They may offer merging or
un-merging as an option.
