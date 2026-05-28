# Rustdoc search

Typing in the search bar instantly searches the available documentation,
matching either the name and path of an item, or a function's approximate
type signature.

## Search By Name

To search by the name of an item (items include modules, types, traits,
functions, and macros), write its name or path. As a special case, the parts
of a path that normally get divided by `::` double colons can instead be
separated by spaces. For example:

  * [`vec new`] and [`vec::new`] both show the function `std::vec::Vec::new`
    as a result.
  * [`vec`], [`vec vec`], [`std::vec`], and [`std::vec::Vec`] all include the struct
    `std::vec::Vec` itself in the results (and all but the last one also
    include the module in the results).

[`vec new`]: ../../std/vec/struct.Vec.html?search=vec%20new&filter-crate=std
[`vec::new`]: ../../std/vec/struct.Vec.html?search=vec::new&filter-crate=std
[`vec`]: ../../std/vec/struct.Vec.html?search=vec&filter-crate=std
[`vec vec`]: ../../std/vec/struct.Vec.html?search=vec%20vec&filter-crate=std
[`std::vec`]: ../../std/vec/struct.Vec.html?search=std::vec&filter-crate=std
[`std::vec::Vec`]: ../../std/vec/struct.Vec.html?search=std::vec::Vec&filter-crate=std
[`std::vec::Vec`]: ../../std/vec/struct.Vec.html?search=std::vec::Vec&filter-crate=std

As a quick way to trim down the list of results, there's a drop-down selector
below the search input, labeled "Results in \[std\]". Clicking it can change
which crate is being searched.

Rustdoc uses a fuzzy matching function that can tolerate typos for this,
though it's based on the length of the name that's typed in, so a good example
of how this works would be [`HahsMap`]. To avoid this, wrap the item in quotes,
searching for `"HahsMap"` (in this example, no results will be returned).

[`HahsMap`]: ../../std/collections/struct.HashMap.html?search=HahsMap&filter-crate=std

### Tabs in the Search By Name interface

In fact, using [`HahsMap`] again as the example, it tells you that you're
using "In Names" by default, but also lists two other tabs below the crate
drop-down: "In Parameters" and "In Return Types".

These two tabs are lists of functions, defined on the closest matching type
to the search (for `HahsMap`, it loudly auto-corrects to `hashmap`). This
auto-correct only kicks in if nothing is found that matches the literal.

These tabs are not just methods. For example, searching the alloc crate for
[`Layout`] also lists functions that accept layouts even though they're
methods on the allocator or free functions.

[`Layout`]: ../../alloc/index.html?search=Layout&filter-crate=alloc

## Searching By Type Signature

If you know more specifically what the function you want to look at does,
or you want to know how to get from one type to another,
Rustdoc can search by more than one type at once in the parameters and return
value. Multiple parameters are separated by `,` commas, and the return value
is written with after a `->` arrow.

Before describing the syntax in more detail, here's a few sample searches of
the standard library and functions that are included in the results list:

| Query | Results |
|-------|---------|
| [`usize -> vec`][] | `slice::repeat` and `Vec::with_capacity` |
| [`vec, vec -> bool`][] | `Vec::eq` |
| [`option<T>, fnonce -> option<U>`][] | `Option::map` and `Option::and_then` |
| [`option<T>, (fnonce (T) -> bool) -> option<T>`][optionfilter] | `Option::filter` |
| [`option<T>, (T -> bool) -> option<T>`][optionfilter2] | `Option::filter` |
| [`option -> default`][] | `Option::unwrap_or_default` |
| [`stdout, [u8]`][stdoutu8] | `Stdout::write` |
| [`any -> !`][] | `panic::panic_any` |
| [`vec::intoiter<T> -> [T]`][iterasslice] | `IntoIter::as_slice` and `IntoIter::next_chunk` |
| [`iterator<T>, fnmut -> T`][iterreduce] | `Iterator::reduce` and `Iterator::find` |

[`usize -> vec`]: ../../std/vec/struct.Vec.html?search=usize%20-%3E%20vec&filter-crate=std
[`vec, vec -> bool`]: ../../std/vec/struct.Vec.html?search=vec,%20vec%20-%3E%20bool&filter-crate=std
[`option<T>, fnonce -> option<U>`]: ../../std/vec/struct.Vec.html?search=option<T>%2C%20fnonce%20->%20option<U>&filter-crate=std
[optionfilter]: ../../std/vec/struct.Vec.html?search=option<T>%2C+(fnonce+(T)+->+bool)+->+option<T>&filter-crate=std
[optionfilter2]: ../../std/vec/struct.Vec.html?search=option<T>%2C+(T+->+bool)+->+option<T>&filter-crate=std
[`option -> default`]: ../../std/vec/struct.Vec.html?search=option%20-%3E%20default&filter-crate=std
[`any -> !`]: ../../std/vec/struct.Vec.html?search=any%20-%3E%20!&filter-crate=std
[stdoutu8]: ../../std/vec/struct.Vec.html?search=stdout%2C%20[u8]&filter-crate=std
[iterasslice]: ../../std/vec/struct.Vec.html?search=vec%3A%3Aintoiter<T>%20->%20[T]&filter-crate=std
[iterreduce]: ../../std/index.html?search=iterator<T>%2C%20fnmut%20->%20T&filter-crate=std

### Non-functions in type-based search
Certain items that are not functions are treated as though they
were a semantically equivalent function.

For example, struct fields are treated as though they were getter methods.
This means that a search for `CpuidResult -> u32` will show
the `CpuidResult::eax` field in the results.

Additionally, `const` and `static` items are treated as nullary functions,
so `-> u32` will match `u32::MAX`.

### How type-based search works

In a complex type-based search, Rustdoc always treats every item's name as literal.
If a name is used and nothing in the docs matches the individual item, such as
a typo-ed [`uize -> vec`][] search, the item `uize` is treated as a generic
type parameter (resulting in `vec::from` and other generic vec constructors).

[`uize -> vec`]: ../../std/vec/struct.Vec.html?search=uize%20-%3E%20vec&filter-crate=std

After deciding which items are type parameters and which are actual types, it
then searches by matching up the function parameters (written before the `->`)
and the return types (written after the `->`). Type matching is order-agnostic,
and allows items to be left out of the query, but items that are present in the
query must be present in the function for it to match. The `self` parameter is
treated the same as any other parameter, and `Self` is resolved to the
underlying type's name.

Function signature searches can query generics, wrapped in angle brackets, and
traits will be normalized like types in the search engine if no type parameters
match them. For example, a function with the signature
`fn my_function<I: Iterator<Item=u32>>(input: I) -> usize`
can be matched with the following queries:

* `Iterator<Item=u32> -> usize`
* `Iterator<u32> -> usize` (you can leave out the `Item=` part)
* `Iterator -> usize` (you can leave out iterator's generic entirely)
* `T -> usize` (you can match with a generic parameter)

Each of the above queries is progressively looser, except the last one
would not match `dyn Iterator`, since that's not a type parameter.

If a bound has multiple associated types, specifying the name allows you to
pick which one gets matched. If no name is specified, then the query will
match of any of them. For example,

```rust
pub trait MyTrait {
    type First;
    type Second;
}

/// This function can be found using the following search queries:
///
///     MyTrait<First=u8, Second=u32> -> bool
///     MyTrait<Second=u32> -> bool
///
/// The following queries, however, will *not* match it:
///
///     MyTrait<First=u32> -> bool
///     MyTrait<u32, u32> -> bool
///     MyTrait<u32, First=u8> -> bool
///     MyTrait<u32, u8> -> bool
pub fn my_fn(x: impl MyTrait<First=u8, Second=u32>) -> bool { true }
```

Function parameters are order-agnostic, but sensitive to nesting
and number of matches. For example, a function with the signature
`fn read_all(&mut self: impl Read) -> Result<Vec<u8>, Error>`
will match these queries:

* `&mut Read -> Result<Vec<u8>, Error>`
* `Read -> Result<Vec<u8>, Error>`
* `Read -> Result<Vec<u8>>`
* `Read -> Vec<u8>`

But it *does not* match `Result<Vec, u8>` or `Result<u8<Vec>>`,
because those are nested incorrectly, and it does not match
`Result<Error, Vec<u8>>` or `Result<Error>`, because those are
in the wrong order. It also does not match `Read -> u8`, because
only [certain generic wrapper types] can be left out, and `Vec` isn't
one of them.

[certain generic wrapper types]: #wrappers-that-can-be-omitted

To search for a function that accepts a function as a parameter,
like `Iterator::all`, wrap the nested signature in parenthesis,
as in [`Iterator<T>, (T -> bool) -> bool`][iterator-all].
You can also search for a specific closure trait,
such as `Iterator<T>, (FnMut(T) -> bool) -> bool`,
but you need to know which one you want.

[iterator-all]: ../../std/vec/struct.Vec.html?search=Iterator<T>%2C+(T+->+bool)+->+bool&filter-crate=std

### Wrappers that can be omitted

* References
* Box
* Rc
* Arc
* Option
* Result
* From
* Into
* Future

### Primitives with Special Syntax

| Shorthand        | Explicit names                                    |
| ---------------- | ------------------------------------------------- |
| `&`              | `primitive:reference`                             |
| `&T`             | `primitive:reference<T>`                          |
| `&mut`           | `primitive:reference<keyword:mut>`                |
| `&mut T`         | `primitive:reference<keyword:mut, T>`             |
| `[]`             | `primitive:slice` and/or `primitive:array`        |
| `[T]`            | `primitive:slice<T>` and/or `primitive:array<T>`  |
| `()`             | `primitive:unit` and/or `primitive:tuple`         |
| `(T)`            | `T`                                               |
| `(T,)`           | `primitive:tuple<T>`                              |
| `!`              | `primitive:never`                                 |
| `(T, U -> V, W)` | `fn(T, U) -> (V, W)`, `Fn`, `FnMut`, and `FnOnce` |

When searching for `[]`, Rustdoc will return search results with either slices
or arrays. If you know which one you want, you can force it to return results
for `primitive:slice` or `primitive:array` using the explicit name syntax.
Empty square brackets, `[]`, will match any slice or array regardless of what
it contains, or an item type can be provided, such as `[u8]` or `[T]`, to
explicitly find functions that operate on byte slices or generic slices,
respectively.

A single type expression wrapped in parens is the same as that type expression,
since parens act as the grouping operator. If they're empty, though, they will
match both `unit` and `tuple`, and if there's more than one type (or a trailing
or leading comma) it is the same as `primitive:tuple<...>`.

However, since items can be left out of the query, `(T)` will still return
results for types that match tuples, even though it also matches the type on
its own. That is, `(u32)` matches `(u32,)` for the exact same reason that it
also matches `Result<u32, Error>`.

The `->` operator has lower precedence than comma. If it's not wrapped
in brackets, it delimits the return value for the function being searched for.
To search for functions that take functions as parameters, use parenthesis.

### Limitations and quirks of type-based search

Type-based search is still a buggy, experimental, work-in-progress feature.
Most of these limitations should be addressed in future version of Rustdoc.

  * There's no way to write trait constraints on generic parameters.
    You can name traits directly, and if there's a type parameter
    with that bound, it'll match, but `option<T> -> T where T: Default`
    cannot be precisely searched for (use `option<Default> -> Default`).

  * Supertraits, type aliases, and Deref are all ignored. Search mostly
    operates on type signatures *as written*, and not as they are
    represented within the compiler.

  * Type parameters match type parameters, such that `Option<A>` matches
    `Option<T>`, but never match concrete types in function signatures.
    A trait named as if it were a type, such as `Option<Read>`, will match
    a type parameter constrained by that trait, such as
    `Option<T> where T: Read`, as well as matching `dyn Trait` and
    `impl Trait`.

  * `impl Trait` in argument position is treated exactly like a type
    parameter, but in return position it will not match type parameters.

  * Any type named in a complex type-based search will be assumed to be a
    type parameter if nothing matching the name exactly is found. If you
    want to force a type parameter, write `generic:T` and it will be used
    as a type parameter even if a matching name is found. If you know
    that you don't want a type parameter, you can force it to match
    something else by giving it a different prefix like `struct:T`.

  * Searching for lifetimes is not supported.

  * It's impossible to search based on the length of an array.

## Item filtering

Names in the search interface can be prefixed with an item type followed by a
colon (such as `mod:`) to restrict the results to just that kind of item. Also,
searching for `println!` will search for a macro named `println`, just like
searching for `macro:println` does. The complete list of available filters is
given under the <kbd>?</kbd> Help area, and in the detailed syntax below.

Item filters can be used in both name-based and type signature-based searches.

## Search query syntax

```text
ident = *(ALPHA / DIGIT / "_")
path = ident *(DOUBLE-COLON ident) [BANG]
slice-like = OPEN-SQUARE-BRACKET [ nonempty-arg-list ] CLOSE-SQUARE-BRACKET
tuple-like = OPEN-PAREN [ nonempty-arg-list ] CLOSE-PAREN
borrow-ref = AMP *WS [MUT] *WS [arg]
arg = [type-filter *WS COLON *WS] (path [generics] / slice-like / tuple-like / borrow-ref)
type-sep = COMMA/WS *(COMMA/WS)
nonempty-arg-list = *(type-sep) arg *(type-sep arg) *(type-sep) [ return-args ]
generic-arg-list = *(type-sep) arg [ EQUAL arg ] *(type-sep arg [ EQUAL arg ]) *(type-sep)
normal-generics = OPEN-ANGLE-BRACKET [ generic-arg-list ] *(type-sep)
            CLOSE-ANGLE-BRACKET
fn-like-generics = OPEN-PAREN [ nonempty-arg-list ] CLOSE-PAREN [ RETURN-ARROW arg ]
generics = normal-generics / fn-like-generics
return-args = RETURN-ARROW *(type-sep) nonempty-arg-list

exact-search = [type-filter *WS COLON] [ RETURN-ARROW ] *WS QUOTE ident QUOTE [ generics ]
type-search = [ nonempty-arg-list ]

query = *WS (exact-search / type-search) *WS

type-filter = (
    "mod" /
    "externcrate" /
    "import" /
    "struct" /
    "enum" /
    "fn" /
    "type" /
    "static" /
    "trait" /
    "impl" /
    "tymethod" /
    "method" /
    "structfield" /
    "variant" /
    "macro" /
    "primitive" /
    "associatedtype" /
    "constant" /
    "associatedconstant" /
    "union" /
    "foreigntype" /
    "keyword" /
    "existential" /
    "attr" /
    "derive" /
    "traitalias" /
    "generic")

OPEN-ANGLE-BRACKET = "<"
CLOSE-ANGLE-BRACKET = ">"
OPEN-SQUARE-BRACKET = "["
CLOSE-SQUARE-BRACKET = "]"
OPEN-PAREN = "("
CLOSE-PAREN = ")"
COLON = ":"
DOUBLE-COLON = "::"
QUOTE = %x22
COMMA = ","
RETURN-ARROW = "->"
EQUAL = "="
BANG = "!"
AMP = "&"
MUT = "mut"

ALPHA = %x41-5A / %x61-7A ; A-Z / a-z
DIGIT = %x30-39
WS = %x09 / " "
```
