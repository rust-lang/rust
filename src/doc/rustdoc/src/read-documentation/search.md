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

## Searching By Type Signature for functions

If you know more specifically what the function you want to look at does,
Rustdoc can search by more than one type at once in the parameters and return
value. Multiple parameters are separated by `,` commas, and the return value
is written with after a `->` arrow.

Before describing the syntax in more detail, here's a few sample searches of
the standard library and functions that are included in the results list:

| Query | Results |
|-------|--------|
| [`usize -> vec`][] | `slice::repeat` and `Vec::with_capacity` |
| [`vec, vec -> bool`][] | `Vec::eq` |
| [`option<T>, fnonce -> option<U>`][] | `Option::map` and `Option::and_then` |
| [`option<T>, fnonce -> option<T>`][] | `Option::filter` and `Option::inspect` |
| [`option -> default`][] | `Option::unwrap_or_default` |
| [`stdout, [u8]`][stdoutu8] | `Stdout::write` |
| [`any -> !`][] | `panic::panic_any` |
| [`vec::intoiter<T> -> [T]`][iterasslice] | `IntoIter::as_slice` and `IntoIter::next_chunk` |

[`usize -> vec`]: ../../std/vec/struct.Vec.html?search=usize%20-%3E%20vec&filter-crate=std
[`vec, vec -> bool`]: ../../std/vec/struct.Vec.html?search=vec,%20vec%20-%3E%20bool&filter-crate=std
[`option<T>, fnonce -> option<U>`]: ../../std/vec/struct.Vec.html?search=option<T>%2C%20fnonce%20->%20option<U>&filter-crate=std
[`option<T>, fnonce -> option<T>`]: ../../std/vec/struct.Vec.html?search=option<T>%2C%20fnonce%20->%20option<T>&filter-crate=std
[`option -> default`]: ../../std/vec/struct.Vec.html?search=option%20-%3E%20default&filter-crate=std
[`any -> !`]: ../../std/vec/struct.Vec.html?search=any%20-%3E%20!&filter-crate=std
[stdoutu8]: ../../std/vec/struct.Vec.html?search=stdout%2C%20[u8]&filter-crate=std
[iterasslice]: ../../std/vec/struct.Vec.html?search=vec%3A%3Aintoiter<T>%20->%20[T]&filter-crate=std

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
query must be present in the function for it to match.

Function signature searches can query generics, wrapped in angle brackets, and
traits will be normalized like types in the search engine if no type parameters
match them. For example, a function with the signature
`fn my_function<I: Iterator<Item=u32>>(input: I) -> usize`
can be matched with the following queries:

* `Iterator<u32> -> usize`
* `Iterator -> usize`

Generics and function parameters are order-agnostic, but sensitive to nesting
and number of matches. For example, a function with the signature
`fn read_all(&mut self: impl Read) -> Result<Vec<u8>, Error>`
will match these queries:

* `Read -> Result<Vec<u8>, Error>`
* `Read -> Result<Error, Vec>`
* `Read -> Result<Vec<u8>>`

But it *does not* match `Result<Vec, u8>` or `Result<u8<Vec>>`.

Function signature searches also support arrays and slices. The explicit name
`primitive:slice<u8>` and `primitive:array<u8>` can be used to match a slice
or array of bytes, while square brackets `[u8]` will match either one. Empty
square brackets, `[]`, will match any slice or array regardless of what
it contains, while a slice with a type parameter, like `[T]`, will only match
functions that actually operate on generic slices.

### Limitations and quirks of type-based search

Type-based search is still a buggy, experimental, work-in-progress feature.
Most of these limitations should be addressed in future version of Rustdoc.

  * There's no way to write trait constraints on generic parameters.
    You can name traits directly, and if there's a type parameter
    with that bound, it'll match, but `option<T> -> T where T: Default`
    cannot be precisely searched for (use `option<Default> -> Default`).

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

  * It's impossible to search for references, pointers, or tuples. The
    wrapped types can be searched for, so a function that takes `&File` can
    be found with `File`, but you'll get a parse error when typing an `&`
    into the search field. Similarly, `Option<(T, U)>` can be matched with
    `Option<T, U>`, but `(` will give a parse error.

  * Searching for lifetimes is not supported.

  * It's impossible to search for closures based on their parameters or
    return values.

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
path = ident *(DOUBLE-COLON ident) [!]
slice = OPEN-SQUARE-BRACKET [ nonempty-arg-list ] CLOSE-SQUARE-BRACKET
arg = [type-filter *WS COLON *WS] (path [generics] / slice / [!])
type-sep = COMMA/WS *(COMMA/WS)
nonempty-arg-list = *(type-sep) arg *(type-sep arg) *(type-sep)
generics = OPEN-ANGLE-BRACKET [ nonempty-arg-list ] *(type-sep)
            CLOSE-ANGLE-BRACKET
return-args = RETURN-ARROW *(type-sep) nonempty-arg-list

exact-search = [type-filter *WS COLON] [ RETURN-ARROW ] *WS QUOTE ident QUOTE [ generics ]
type-search = [ nonempty-arg-list ] [ return-args ]

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
COLON = ":"
DOUBLE-COLON = "::"
QUOTE = %x22
COMMA = ","
RETURN-ARROW = "->"

ALPHA = %x41-5A / %x61-7A ; A-Z / a-z
DIGIT = %x30-39
WS = %x09 / " "
```
