# Generics

## Generic functions

Throughout this tutorial, I've been defining functions like `for_rev`
that act only on integers. It is 2012, and we no longer expect to be
defining such functions again and again for every type they apply to.
Thus, Rust allows functions and datatypes to have type parameters.

    fn for_rev<T>(v: [T], act: block(T)) {
        let i = vec::len(v);
        while i > 0u {
            i -= 1u;
            act(v[i]);
        }
    }
    
    fn map<T, U>(v: [T], f: block(T) -> U) -> [U] {
        let acc = [];
        for elt in v { acc += [f(elt)]; }
        ret acc;
    }

When defined in this way, these functions can be applied to any type
of vector, as long as the type of the block's argument and the type of
the vector's content agree with each other.

Inside a parameterized (generic) function, the names of the type
parameters (capitalized by convention) stand for opaque types. You
can't look inside them, but you can pass them around.

## Generic datatypes

Generic `type` and `enum` declarations follow the same pattern:

    type circular_buf<T> = {start: uint,
                            end: uint,
                            buf: [mutable T]};
    
    enum option<T> { some(T); none; }

You can then declare a function to take a `circular_buf<u8>` or return
an `option<str>`, or even an `option<T>` if the function itself is
generic.

The `option` type given above exists in the core library as
`option::t`, and is the way Rust programs express the thing that in C
would be a nullable pointer. The nice part is that you have to
explicitly unpack an `option` type, so accidental null pointer
dereferences become impossible.

## Type-inference and generics

Rust's type inferrer works very well with generics, but there are
programs that just can't be typed.

    let n = option::none;
    # n = option::some(1);

If you never do anything else with `n`, the compiler will not be able
to assign a type to it. (The same goes for `[]`, the empty vector.) If
you really want to have such a statement, you'll have to write it like
this:

    let n2: option::t<int> = option::none;
    // or
    let n = option::none::<int>;

Note that, in a value expression, `<` already has a meaning as a
comparison operator, so you'll have to write `::<T>` to explicitly
give a type to a name that denotes a generic value. Fortunately, this
is rarely necessary.

## Polymorphic built-ins

There are two built-in operations that, perhaps surprisingly, act on
values of any type. It was already mentioned earlier that `log` can
take any type of value and output it.

More interesting is that Rust also defines an ordering for values of
all datatypes, and allows you to meaningfully apply comparison
operators (`<`, `>`, `<=`, `>=`, `==`, `!=`) to them. For structural
types, the comparison happens left to right, so `"abc" < "bac"` (but
note that `"bac" < "ác"`, because the ordering acts on UTF-8 sequences
without any sophistication).

## Kinds

<a name="kind"></a>

Perhaps surprisingly, the 'copy' (duplicate) operation is not defined
for all Rust types. Resource types (types with destructors) can not be
copied, and neither can any type whose copying would require copying a
resource (such as records or unique boxes containing a resource).

This complicates handling of generic functions. If you have a type
parameter `T`, can you copy values of that type? In Rust, you can't,
unless you explicitly declare that type parameter to have copyable
'kind'. A kind is a type of type.

    ## ignore
    // This does not compile
    fn head_bad<T>(v: [T]) -> T { v[0] }
    // This does
    fn head<T: copy>(v: [T]) -> T { v[0] }

When instantiating a generic function, you can only instantiate it
with types that fit its kinds. So you could not apply `head` to a
resource type.

Rust has three kinds: 'noncopyable', 'copyable', and 'sendable'. By
default, type parameters are considered to be noncopyable. You can
annotate them with the `copy` keyword to declare them copyable, and
with the `send` keyword to make them sendable.

Sendable types are a subset of copyable types. They are types that do
not contain shared (reference counted) types, which are thus uniquely
owned by the function that owns them, and can be sent over channels to
other tasks. Most of the generic functions in the core `comm` module
take sendable types.

## Generic functions and argument-passing

The previous section mentioned that arguments are passed by pointer or
by value based on their type. There is one situation in which this is
difficult. If you try this program:

    # fn map(f: block(int) -> int, v: [int]) {}
    fn plus1(x: int) -> int { x + 1 }
    map(plus1, [1, 2, 3]);

You will get an error message about argument passing styles
disagreeing. The reason is that generic types are always passed by
pointer, so `map` expects a function that takes its argument by
pointer. The `plus1` you defined, however, uses the default, efficient
way to pass integers, which is by value. To get around this issue, you
have to explicitly mark the arguments to a function that you want to
pass to a generic higher-order function as being passed by pointer,
using the `&&` sigil:

    # fn map<T, U>(f: block(T) -> U, v: [T]) {}
    fn plus1(&&x: int) -> int { x + 1 }
    map(plus1, [1, 2, 3]);

NOTE: This is inconvenient, and we are hoping to get rid of this
restriction in the future.
