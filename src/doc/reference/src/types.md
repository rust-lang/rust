# Types

Every variable, item and value in a Rust program has a type. The _type_ of a
*value* defines the interpretation of the memory holding it.

Built-in types and type-constructors are tightly integrated into the language,
in nontrivial ways that are not possible to emulate in user-defined types.
User-defined types have limited capabilities.

## Primitive types

The primitive types are the following:

* The boolean type `bool` with values `true` and `false`.
* The machine types (integer and floating-point).
* The machine-dependent integer types.
* Arrays
* Tuples
* Slices
* Function pointers

### Machine types

The machine types are the following:

* The unsigned word types `u8`, `u16`, `u32` and `u64`, with values drawn from
  the integer intervals [0, 2^8 - 1], [0, 2^16 - 1], [0, 2^32 - 1] and
  [0, 2^64 - 1] respectively.

* The signed two's complement word types `i8`, `i16`, `i32` and `i64`, with
  values drawn from the integer intervals [-(2^(7)), 2^7 - 1],
  [-(2^(15)), 2^15 - 1], [-(2^(31)), 2^31 - 1], [-(2^(63)), 2^63 - 1]
  respectively.

* The IEEE 754-2008 `binary32` and `binary64` floating-point types: `f32` and
  `f64`, respectively.

### Machine-dependent integer types

The `usize` type is an unsigned integer type with the same number of bits as the
platform's pointer type. It can represent every memory address in the process.

The `isize` type is a signed integer type with the same number of bits as the
platform's pointer type. The theoretical upper bound on object and array size
is the maximum `isize` value. This ensures that `isize` can be used to calculate
differences between pointers into an object or array and can address every byte
within an object along with one byte past the end.

## Textual types

The types `char` and `str` hold textual data.

A value of type `char` is a [Unicode scalar value](
http://www.unicode.org/glossary/#unicode_scalar_value) (i.e. a code point that
is not a surrogate), represented as a 32-bit unsigned word in the 0x0000 to
0xD7FF or 0xE000 to 0x10FFFF range. A `[char]` array is effectively an UCS-4 /
UTF-32 string.

A value of type `str` is a Unicode string, represented as an array of 8-bit
unsigned bytes holding a sequence of UTF-8 code points. Since `str` is of
unknown size, it is not a _first-class_ type, but can only be instantiated
through a pointer type, such as `&str`.

## Tuple types

A tuple *type* is a heterogeneous product of other types, called the *elements*
of the tuple. It has no nominal name and is instead structurally typed.

Tuple types and values are denoted by listing the types or values of their
elements, respectively, in a parenthesized, comma-separated list.

Because tuple elements don't have a name, they can only be accessed by
pattern-matching or by using `N` directly as a field to access the
`N`th element.

An example of a tuple type and its use:

```
type Pair<'a> = (i32, &'a str);
let p: Pair<'static> = (10, "ten");
let (a, b) = p;

assert_eq!(a, 10);
assert_eq!(b, "ten");
assert_eq!(p.0, 10);
assert_eq!(p.1, "ten");
```

For historical reasons and convenience, the tuple type with no elements (`()`)
is often called ‘unit’ or ‘the unit type’.

## Array, and Slice types

Rust has two different types for a list of items:

* `[T; N]`, an 'array'
* `&[T]`, a 'slice'

An array has a fixed size, and can be allocated on either the stack or the
heap.

A slice is a 'view' into an array. It doesn't own the data it points
to, it borrows it.

Examples:

```{rust}
// A stack-allocated array
let array: [i32; 3] = [1, 2, 3];

// A heap-allocated array
let vector: Vec<i32> = vec![1, 2, 3];

// A slice into an array
let slice: &[i32] = &vector[..];
```

As you can see, the `vec!` macro allows you to create a `Vec<T>` easily. The
`vec!` macro is also part of the standard library, rather than the language.

All in-bounds elements of arrays and slices are always initialized, and access
to an array or slice is always bounds-checked.

## Struct types

A `struct` *type* is a heterogeneous product of other types, called the
*fields* of the type.[^structtype]

[^structtype]: `struct` types are analogous to `struct` types in C,
    the *record* types of the ML family,
    or the *struct* types of the Lisp family.

New instances of a `struct` can be constructed with a [struct
expression](expressions.html#struct-expressions).

The memory layout of a `struct` is undefined by default to allow for compiler
optimizations like field reordering, but it can be fixed with the
`#[repr(...)]` attribute. In either case, fields may be given in any order in
a corresponding struct *expression*; the resulting `struct` value will always
have the same memory layout.

The fields of a `struct` may be qualified by [visibility
modifiers](visibility-and-privacy.html), to allow access to data in a
struct outside a module.

A _tuple struct_ type is just like a struct type, except that the fields are
anonymous.

A _unit-like struct_ type is like a struct type, except that it has no
fields. The one value constructed by the associated [struct
expression](expressions.html#struct-expressions) is the only value that inhabits such a
type.

## Enumerated types

An *enumerated type* is a nominal, heterogeneous disjoint union type, denoted
by the name of an [`enum` item](items.html#enumerations). [^enumtype]

[^enumtype]: The `enum` type is analogous to a `data` constructor declaration in
             ML, or a *pick ADT* in Limbo.

An [`enum` item](items.html#enumerations) declares both the type and a number of *variant
constructors*, each of which is independently named and takes an optional tuple
of arguments.

New instances of an `enum` can be constructed by calling one of the variant
constructors, in a [call expression](expressions.html#call-expressions).

Any `enum` value consumes as much memory as the largest variant constructor for
its corresponding `enum` type.

Enum types cannot be denoted *structurally* as types, but must be denoted by
named reference to an [`enum` item](items.html#enumerations).

## Recursive types

Nominal types &mdash; [enumerations](#enumerated-types) and
[structs](#struct-types) &mdash; may be recursive. That is, each `enum`
constructor or `struct` field may refer, directly or indirectly, to the
enclosing `enum` or `struct` type itself. Such recursion has restrictions:

* Recursive types must include a nominal type in the recursion
  (not mere [type definitions](../grammar.html#type-definitions),
   or other structural types such as [arrays](#array-and-slice-types) or [tuples](#tuple-types)).
* A recursive `enum` item must have at least one non-recursive constructor
  (in order to give the recursion a basis case).
* The size of a recursive type must be finite;
  in other words the recursive fields of the type must be [pointer types](#pointer-types).
* Recursive type definitions can cross module boundaries, but not module *visibility* boundaries,
  or crate boundaries (in order to simplify the module system and type checker).

An example of a *recursive* type and its use:

```
enum List<T> {
    Nil,
    Cons(T, Box<List<T>>)
}

let a: List<i32> = List::Cons(7, Box::new(List::Cons(13, Box::new(List::Nil))));
```

## Pointer types

All pointers in Rust are explicit first-class values. They can be copied,
stored into data structs, and returned from functions. There are two
varieties of pointer in Rust:

* References (`&`)
  : These point to memory _owned by some other value_.
    A reference type is written `&type`,
    or `&'a type` when you need to specify an explicit lifetime.
    Copying a reference is a "shallow" operation:
    it involves only copying the pointer itself.
    Releasing a reference has no effect on the value it points to,
    but a reference of a temporary value will keep it alive during the scope
    of the reference itself.

* Raw pointers (`*`)
  : Raw pointers are pointers without safety or liveness guarantees.
    Raw pointers are written as `*const T` or `*mut T`,
    for example `*const i32` means a raw pointer to a 32-bit integer.
    Copying or dropping a raw pointer has no effect on the lifecycle of any
    other value. Dereferencing a raw pointer or converting it to any other
    pointer type is an [`unsafe` operation](unsafe-functions.html).
    Raw pointers are generally discouraged in Rust code;
    they exist to support interoperability with foreign code,
    and writing performance-critical or low-level functions.

The standard library contains additional 'smart pointer' types beyond references
and raw pointers.

## Function types

The function type constructor `fn` forms new function types. A function type
consists of a possibly-empty set of function-type modifiers (such as `unsafe`
or `extern`), a sequence of input types and an output type.

An example of a `fn` type:

```
fn add(x: i32, y: i32) -> i32 {
    x + y
}

let mut x = add(5,7);

type Binop = fn(i32, i32) -> i32;
let bo: Binop = add;
x = bo(5,7);
```

### Function types for specific items

Internal to the compiler, there are also function types that are specific to a particular
function item. In the following snippet, for example, the internal types of the functions
`foo` and `bar` are different, despite the fact that they have the same signature:

```
fn foo() { }
fn bar() { }
```

The types of `foo` and `bar` can both be implicitly coerced to the fn
pointer type `fn()`. There is currently no syntax for unique fn types,
though the compiler will emit a type like `fn() {foo}` in error
messages to indicate "the unique fn type for the function `foo`".

## Closure types

A [lambda expression](expressions.html#lambda-expressions) produces a closure
value with a unique, anonymous type that cannot be written out.

Depending on the requirements of the closure, its type implements one or
more of the closure traits:

* `FnOnce`
  : The closure can be called once. A closure called as `FnOnce`
    can move out values from its environment.

* `FnMut`
  : The closure can be called multiple times as mutable. A closure called as
    `FnMut` can mutate values from its environment. `FnMut` inherits from
    `FnOnce` (i.e. anything implementing `FnMut` also implements `FnOnce`).

* `Fn`
  : The closure can be called multiple times through a shared reference.
    A closure called as `Fn` can neither move out from nor mutate values
    from its environment. `Fn` inherits from `FnMut`, which itself
    inherits from `FnOnce`.


## Trait objects

In Rust, a type like `&SomeTrait` or `Box<SomeTrait>` is called a _trait object_.
Each instance of a trait object includes:

 - a pointer to an instance of a type `T` that implements `SomeTrait`
 - a _virtual method table_, often just called a _vtable_, which contains, for
   each method of `SomeTrait` that `T` implements, a pointer to `T`'s
   implementation (i.e. a function pointer).

The purpose of trait objects is to permit "late binding" of methods. Calling a
method on a trait object results in virtual dispatch at runtime: that is, a
function pointer is loaded from the trait object vtable and invoked indirectly.
The actual implementation for each vtable entry can vary on an object-by-object
basis.

Note that for a trait object to be instantiated, the trait must be
_object-safe_. Object safety rules are defined in [RFC 255].

[RFC 255]: https://github.com/rust-lang/rfcs/blob/master/text/0255-object-safety.md

Given a pointer-typed expression `E` of type `&T` or `Box<T>`, where `T`
implements trait `R`, casting `E` to the corresponding pointer type `&R` or
`Box<R>` results in a value of the _trait object_ `R`. This result is
represented as a pair of pointers: the vtable pointer for the `T`
implementation of `R`, and the pointer value of `E`.

An example of a trait object:

```
trait Printable {
    fn stringify(&self) -> String;
}

impl Printable for i32 {
    fn stringify(&self) -> String { self.to_string() }
}

fn print(a: Box<Printable>) {
    println!("{}", a.stringify());
}

fn main() {
    print(Box::new(10) as Box<Printable>);
}
```

In this example, the trait `Printable` occurs as a trait object in both the
type signature of `print`, and the cast expression in `main`.

### Type parameters

Within the body of an item that has type parameter declarations, the names of
its type parameters are types:

```ignore
fn to_vec<A: Clone>(xs: &[A]) -> Vec<A> {
    if xs.is_empty() {
        return vec![];
    }
    let first: A = xs[0].clone();
    let mut rest: Vec<A> = to_vec(&xs[1..]);
    rest.insert(0, first);
    rest
}
```

Here, `first` has type `A`, referring to `to_vec`'s `A` type parameter; and `rest`
has type `Vec<A>`, a vector with element type `A`.

## Self types

The special type `Self` has a meaning within traits and impls. In a trait definition, it refers
to an implicit type parameter representing the "implementing" type. In an impl,
it is an alias for the implementing type. For example, in:

```
pub trait From<T> {
    fn from(T) -> Self;
}

impl From<i32> for String {
    fn from(x: i32) -> Self {
        x.to_string()
    }
}
```

The notation `Self` in the impl refers to the implementing type: `String`. In another
example:

```
trait Printable {
    fn make_string(&self) -> String;
}

impl Printable for String {
    fn make_string(&self) -> String {
        (*self).clone()
    }
}
```

The notation `&self` is a shorthand for `self: &Self`. In this case,
in the impl, `Self` refers to the value of type `String` that is the
receiver for a call to the method `make_string`.
