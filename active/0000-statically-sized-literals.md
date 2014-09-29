- Start Date: 2014-09-29
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Change the types of array, byte string and string literals to be references to statically sized types.
Introduce strings of fixed size.

# Motivation

Currently byte string and string literals have types `&'static [u8]` and `&'static str`.  
Therefore, although the sizes of the literals are known at compile time, they are erased from their types and inaccessible until runtime.  
This RFC suggests to change the types to `&'static [u8, ..N]` and `&'static str[..N]` respectively.
Additionally this RFC suggests to change the types of array literals from `[T, ..N]` to `&'a [T, ..N]` for consistency and ergonomics.

Today, given the lack of non-type generic parameters and compile time (function) evaluation (CTE), strings of fixed size are not very useful.
But after introduction of CTE the need in compile time string operations will raise rapidly.
Even without CTE but with non-type generic parameters alone fixed size strings can be used in runtime for "heapless" string operations, which are useful in constrained environments or for optimization.  
So the main motivation for changes today is forward compatibility and before 1.0 `str[..N]` can be implemented as marginally as possible to allow the change of the types of string literals.

Examples of use for new literals, that are not possible with old literals:

```
// Today: initialize mutable array with byte string literal
let mut arr: [u8, ..3] = *b"abc";
arr[0] = b'd';

// Future with CTE: compile time string concatenation
static LANG_DIR: str[..5 /*The size should, probably, be inferred*/ ] = *"lang/";
static EN_FILE: str[.._] = LANG_DIR + *"en"; // str[..N] implements Add
static FR_FILE: str[.._] = LANG_DIR + *"fr";

// Future without CTE: runtime "heapless" string concatenation
let DE_FILE = LANG_DIR + *"de"; // Performed at runtime if not optimized
```

# Detailed design

### Proposed changes:

1)  
Change the types of array literals from `[T, ..N]` to `&'a [T, ..N]`.  
Change the types of byte string literals from `&'static [u8]` to `&'static [u8, ..N]`.  
Change the types of string literals form `&'static str` to `&'static str[..N]`.  
2)  
Introduce the missing family of types - strings of fixed size - `str[..N]`.  
`str[..N]` is essentially a `[u8, ..N]` with UTF-8 invariants and, eventually, additional string methods/traits.  
It fills the gap in the vector/string chart:

`Vec<T>` | `String`
---------|--------
`[T, ..N]` | ???
`&[T]`   | `&str`

### Static lifetime

Although all the literals under consideration are similar and are essentially arrays of fixed size, array literals are different from byte string and string literals with regard to lifetimes.  
While byte string and string literals can always be placed into static memory and have static lifetime, array literals can depend on local variables and can't have static lifetime in general case.  
The chosen design potentially allows to trivially enhance *some* array literals with static lifetime in the future to allow use like
```
fn f() -> &'static [int] {
    [1, 2, 3]
}
```
, but this RFC doesn't propose such an enhancement.

### Usage statistics for array literals

Array literals can be used both as slices, when a view to array is sufficient to perform the task, and as values when arrays themselves should be copied or modified.  
The exact estimation of the frequencies of both uses is problematic, but some regex search in the Rust codebase gives the next statistics:  
In approximately *70%* of cases array literals are used as slices (explicit `&` on array literals, immutable bindings).  
In approximately *20%* of cases array literals are used as values (initialization of struct fields, mutable bindings,   boxes).  
In the rest *10%* of cases the usage is unclear.  

So, in most cases the change to the types of array literals will lead to shorter notation.

### Backward compatibility

No code using the literals as slices is broken, DST coercions `&[T, ..N] -> &[T], &str[..N] -> &str` do all the job for compatibility.
```
fn f(arg: &str) {}
f("Hello"); // DST coercion

static GOODBYE: &'static str = "Goodbye"; // DST coercion

fn main() {
    let s = "Hello";
    fn f(arg: &str) {}
    f(s); // No breakage, DST coercion
}

fn g(arg: &[int]) {}
g([1i, 2, 3]); // DST coercion &[int, ..3] -> &[int]
```
Unfortunately, autocoercions from arrays of fixed size to slices was prohibited too soon and a lot of array literals like `[1, 2, 3]` were changed to `&[1, 2, 3]`. These changes have to be reverted (but the prohibition of autocoercions should stay in place).

Code using array literals as values is broken, but can be fixed easily.
```
// Array as a struct field
struct S {
    arr: [int, ..3],
}

let s = S { arr: [1, 2, 3] }; // Have to be changed to let s = S { arr: *[1, 2, 3] };

// Mutable array
let mut a = [1i, 2, 3]; // Have to be changed to let mut a = *[1i, 2, 3];
```
This explicit dereference has some benefits - you have to opt-in to use arrays as values and potentially costly array copies become a bit more visible and searchable.  
Anyway, array literals are less frequently used as values (see the statistics), but more often as slices.

### Precedents

C and C++ string literals are lvalue `char` arrays of fixed size with static duration.  
C++ library proposal for strings of fixed size ([link][1]), the paper also contains some discussion and motivation.

# Drawbacks

Some breakage for array literals. See "Backward compatibility" section.

# Alternatives

The alternative design is to make the literals the values and not the references.

### The changes

1)  
Keep the types of array literals as `[T, ..N]`.  
Change the types of byte literals from `&'static [u8]` to `[u8, ..N]`.  
Change the types of string literals form `&'static str` to to `str[..N]`.  
2)  
Introduce the missing family of types - strings of fixed size - `str[..N]`.  
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
static LANG_DIR: str[.._] = "lang/";
static EN_FILE: str[.._] = LANG_DIR + "en"; // str[..N] implements Add
static FR_FILE: str[.._] = LANG_DIR + "fr";

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

// Future: str[..N] is added, CTE is added, but the literal types remain old
let mut arr: [u8, ..3] = b"abc".to_fixed(); // Have to use a conversion method
arr[0] = b'd';

static LANG_DIR: str[.._] = "lang/".to_fixed(); // Have to use a conversion method
static EN_FILE: str[.._] = LANG_DIR + "en".to_fixed();
static FR_FILE: str[.._] = LANG_DIR + "fr".to_fixed();

// Bad future: str[..N] is not added
// "Heapless"/compile-time string operations aren't possible, or performed with "magic" like extended concat! or recursive macros.
```
Note, that in the "Future" scenario the return *type* of `to_fixed` depends on the *value* of `self`, so it requires sufficiently advanced CTE, for example C++14 with its powerful `constexpr` machinery still doesn't allow to write such a function.

# Unresolved questions

If `str` is moved from core language to the library and implemented as a wrapper around u8 array, then strings of fixed size will require additional attention. Moreover, the changes to string literals should, probably, be applied after this move.

Assume we implemented `str` like this:
```
struct StrImpl<T> { underlying_array: T }

type str = StrImpl<[u8]>;
type<N: uint> str_of_fixed_size_bikeshed<N> = StrImpl<[u8, ..N]>; // Non-type generic parameters are required
```
Then `&str_of_fixed_size_bikeshed<N>` (the type of string literals) should somehow autocoerce to `&str` and this coercion is not covered by the current rules.

One possible solution is to make `str` a "not-so-smart" pointer to unsized type and not the unsized type itself.
```
struct StrImplVal<T> { underlying_array: T }
struct StrImplRef<'a, T> { ref_: &'a StrImplVal<T> }

type<'a> str<'a> = StrImplRef<'a, [u8]>;
type<'a, N: uint> ref_to_str_of_fixed_size_bikeshed<'a, N> = StrImplRef<'a, [u8, ..N]>; // Non-type generic parameters are required
type<N: uint> str_of_fixed_size_bikeshed<N> = StrImplVal<[u8, ..N]>; // Non-type generic parameters are required
```
In this case string literals have types `ref_to_str_of_fixed_size_bikeshed<'static, N>` and strings of fixed size have types `str_of_fixed_size_bikeshed<N>`.  
And the coercion from `ref_to_str_of_fixed_size_bikeshed<'a, N>` to `str<'a>` (`StrImplRef<'a, [u8, ..N]> -> StrImplRef<'a, [u8]>`) is an usual DST coercion.  
And dereference on `ref_to_str_of_fixed_size_bikeshed<'a, N>` should return `&'a str_of_fixed_size_bikeshed<N>`.  
And every `&'a str` has to be rewritten as `str<'a>` (and `&str` as `str`), which is a terribly backward incompatible change (but automatically fixable).  
I suppose this change to `str` may be useful by itself and can be proposed as a separate RFC.

 [1]: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4121.pdf
