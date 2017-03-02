- Start Date: 2014-09-29
- RFC PR: [rust-lang/rfcs#339](https://github.com/rust-lang/rfcs/pull/339)
- Rust Issue: [rust-lang/rust#18465](https://github.com/rust-lang/rust/issues/18465)

# Summary

Change the types of byte string literals to be references to statically sized types.
Ensure the same change can be performed backward compatibly for string literals in the future.

# Motivation

Currently byte string and string literals have types `&'static [u8]` and `&'static str`.
Therefore, although the sizes of the literals are known at compile time, they are erased from their types and inaccessible until runtime.
This RFC suggests to change the type of byte string literals to `&'static [u8, ..N]`.
In addition this RFC suggest not to introduce any changes to `str` or string literals, that would prevent a backward compatible addition of strings of fixed size `FixedString<N>` (the name FixedString in this RFC is a placeholder and is open for bikeshedding) and the change of the type of string literals to `&'static FixedString<N>` in the future.

`FixedString<N>` is essentially a `[u8, ..N]` with UTF-8 invariants and additional string methods/traits.
It fills the gap in the vector/string chart:

`Vec<T>` | `String`
---------|--------
`[T, ..N]` | ???
`&[T]`   | `&str`

Today, given the lack of non-type generic parameters and compile time (function) evaluation (CTE), strings of fixed size are not very useful.
But after introduction of CTE the need in compile time string operations will raise rapidly.
Even without CTE but with non-type generic parameters alone fixed size strings can be used in runtime for "heapless" string operations, which are useful in constrained environments or for optimization. So the main motivation for changes today is forward compatibility.

Examples of use for new literals, that are not possible with old literals:

```
// Today: initialize mutable array with byte string literal
let mut arr: [u8, ..3] = *b"abc";
arr[0] = b'd';

// Future with CTE: compile time string concatenation
static LANG_DIR: FixedString<5 /*The size should, probably, be inferred*/> = *"lang/";
static EN_FILE: FixedString<_> = LANG_DIR + *"en"; // FixedString<N> implements Add
static FR_FILE: FixedString<_> = LANG_DIR + *"fr";

// Future without CTE: runtime "heapless" string concatenation
let DE_FILE = LANG_DIR + *"de"; // Performed at runtime if not optimized
```

# Detailed design

Change the type of byte string literals from `&'static [u8]` to `&'static [u8, ..N]`.
Leave the door open for a backward compatible change of the type of string literals from `&'static str` to `&'static FixedString<N>`.

### Strings of fixed size

If `str` is moved to the library today, then strings of fixed size can be implemented like this:
```
struct str<Sized? T = [u8]>(T);
```
Then string literals will have types `&'static str<[u8, ..N]>`.

Drawbacks of this approach include unnecessary exposition of the implementation - underlying sized or unsized arrays `[u8]`/`[u8, ..N]` and generic parameter `T`.
The key requirement here is the autocoercion from reference to fixed string to string slice an we are unable to meet it now without exposing the implementation.

In the future, after gaining the ability to parameterize on integers, strings of fixed size could be implemented in a better way:
```
struct __StrImpl<Sized? T>(T); // private

pub type str = __StrImpl<[u8]>; // unsized referent of string slice `&str`, public
pub type FixedString<const N: uint> = __StrImpl<[u8, ..N]>; // string of fixed size, public

// &FixedString<N> -> &str : OK, including &'static FixedString<N> -> &'static str for string literals
```
So, we don't propose to make these changes today and suggest to wait until generic parameterization on integers is added to the language.

### Precedents

C and C++ string literals are lvalue `char` arrays of fixed size with static duration.
C++ library proposal for strings of fixed size ([link][1]), the paper also contains some discussion and motivation.

# Rejected alternatives and discussion

## Array literals

The types of array literals potentially can be changed from `[T, ..N]` to `&'a [T, ..N]` for consistency with the other literals and ergonomics.
The major blocker for this change is the inability to move out from a dereferenced array literal if `T` is not `Copy`.
```
let mut a = *[box 1i, box 2, box 3]; // Wouldn't work without special-casing of array literals with regard to moving out from dereferenced borrowed pointer
```
Despite that array literals as references have better usability, possible `static`ness and consistency with other literals.

### Usage statistics for array literals

Array literals can be used both as slices, when a view to array is sufficient to perform the task, and as values when arrays themselves should be copied or modified.
The exact estimation of the frequencies of both uses is problematic, but some regex search in the Rust codebase gives the next statistics:
In approximately *70%* of cases array literals are used as slices (explicit `&` on array literals, immutable bindings).
In approximately *20%* of cases array literals are used as values (initialization of struct fields, mutable bindings,   boxes).
In the rest *10%* of cases the usage is unclear.

So, in most cases the change to the types of array literals will lead to shorter notation.

### Static lifetime

Although all the literals under consideration are similar and are essentially arrays of fixed size, array literals are different from byte string and string literals with regard to lifetimes.
While byte string and string literals can always be placed into static memory and have static lifetime, array literals can depend on local variables and can't have static lifetime in general case.
The chosen design potentially allows to trivially enhance *some* array literals with static lifetime in the future to allow use like
```
fn f() -> &'static [int] {
    [1, 2, 3]
}
```

## Alternatives

The alternative design is to make the literals the values and not the references.

### The changes

1)
Keep the types of array literals as `[T, ..N]`.
Change the types of byte literals from `&'static [u8]` to `[u8, ..N]`.
Change the types of string literals form `&'static str` to to `FixedString<N>`.
2)
Introduce the missing family of types - strings of fixed size - `FixedString<N>`.
...
3)
Add the autocoercion of array *literals* (not arrays of fixed size in general) to slices.
Add the autocoercion of new byte literals to slices.
Add the autocoercion of new string literals to slices.
Non-literal arrays and strings do not autocoerce to slices, in accordance with the general agreements on explicitness.
4)
Make string and byte literals lvalues with static lifetime.

Examples of use:
```
// Today: initialize mutable array with literal
let mut arr: [u8, ..3] = b"abc";
arr[0] = b'd';

// Future with CTE: compile time string concatenation
static LANG_DIR: FixedString<_> = "lang/";
static EN_FILE: FixedString<_> = LANG_DIR + "en"; // FixedString<N> implements Add
static FR_FILE: FixedString<_> = LANG_DIR + "fr";

// Future without CTE: runtime "heapless" string concatenation
let DE_FILE = LANG_DIR + "de"; // Performed at runtime if not optimized
```

### Drawbacks of the alternative design

Special rules about (byte) string literals being static lvalues add a bit of unnecessary complexity to the specification.

In theory `let s = "abcd";` copies the string from static memory to stack, but the copy is unobservable an can, probably, be elided in most cases.

The set of additional autocoercions has to exist for ergonomic purpose (and for backward compatibility).
Writing something like:
```
fn f(arg: &str) {}
f("Hello"[]);
f(&"Hello");
```
for all literals would be just unacceptable.

Minor breakage:
```
fn main() {
    let s = "Hello";
    fn f(arg: &str) {}
    f(s); // Will require explicit slicing f(s[]) or implicit DST coersion from reference f(&s)
}
```

### Status quo

Status quo (or partial application of the changes) is always an alternative.

### Drawbacks of status quo

Examples:
```
// Today: can't use byte string literals in some cases
let mut arr: [u8, ..3] = [b'a', b'b', b'c']; // Have to use array literals
arr[0] = b'd';

// Future: FixedString<N> is added, CTE is added, but the literal types remain old
let mut arr: [u8, ..3] = b"abc".to_fixed(); // Have to use a conversion method
arr[0] = b'd';

static LANG_DIR: FixedString<_> = "lang/".to_fixed(); // Have to use a conversion method
static EN_FILE: FixedString<_> = LANG_DIR + "en".to_fixed();
static FR_FILE: FixedString<_> = LANG_DIR + "fr".to_fixed();

// Bad future: FixedString<N> is not added
// "Heapless"/compile-time string operations aren't possible, or performed with "magic" like extended concat! or recursive macros.
```
Note, that in the "Future" scenario the return *type* of `to_fixed` depends on the *value* of `self`, so it requires sufficiently advanced CTE, for example C++14 with its powerful `constexpr` machinery still doesn't allow to write such a function.

# Drawbacks

None.

# Unresolved questions

None.

 [1]: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4121.pdf
