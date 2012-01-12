# Datatypes

Rust datatypes are, by default, immutable. The core datatypes of Rust
are structural records and 'enums' (tagged unions, algebraic data
types).

    type point = {x: float, y: float};
    enum shape {
        circle(point, float);
        rectangle(point, point);
    }
    let my_shape = circle({x: 0.0, y: 0.0}, 10.0);

## Records

Rust record types are written `{field1: TYPE, field2: TYPE [, ...]}`,
and record literals are written in the same way, but with expressions
instead of types. They are quite similar to C structs, and even laid
out the same way in memory (so you can read from a Rust struct in C,
and vice-versa).

The dot operator is used to access record fields (`mypoint.x`).

Fields that you want to mutate must be explicitly marked as such. For
example...

    type stack = {content: [int], mutable head: uint};

With such a type, you can do `mystack.head += 1u`. If `mutable` were
omitted from the type, such an assignment would result in a type
error.

To 'update' an immutable record, you use functional record update
syntax, by ending a record literal with the keyword `with`:

    let oldpoint = {x: 10f, y: 20f};
    let newpoint = {x: 0f with oldpoint};
    assert newpoint == {x: 0f, y: 20f};

This will create a new struct, copying all the fields from `oldpoint`
into it, except for the ones that are explicitly set in the literal.

Rust record types are *structural*. This means that `{x: float, y:
float}` is not just a way to define a new type, but is the actual name
of the type. Record types can be used without first defining them. If
module A defines `type point = {x: float, y: float}`, and module B,
without knowing anything about A, defines a function that returns an
`{x: float, y: float}`, you can use that return value as a `point` in
module A. (Remember that `type` defines an additional name for a type,
not an actual new type.)

## Record patterns

Records can be destructured on in `alt` patterns. The basic syntax is
`{fieldname: pattern, ...}`, but the pattern for a field can be
omitted as a shorthand for simply binding the variable with the same
name as the field.

    # let mypoint = {x: 0f, y: 0f};
    alt mypoint {
        {x: 0f, y: y_name} { /* Provide sub-patterns for fields */ }
        {x, y}             { /* Simply bind the fields */ }
    }

The field names of a record do not have to appear in a pattern in the
same order they appear in the type. When you are not interested in all
the fields of a record, a record pattern may end with `, _` (as in
`{field1, _}`) to indicate that you're ignoring all other fields.

## Enums

Enums are datatypes that have several different representations. For
example, the type shown earlier:

    # type point = {x: float, y: float};
    enum shape {
        circle(point, float);
        rectangle(point, point);
    }

A value of this type is either a circle¸ in which case it contains a
point record and a float, or a rectangle, in which case it contains
two point records. The run-time representation of such a value
includes an identifier of the actual form that it holds, much like the
'tagged union' pattern in C, but with better ergonomics.

The above declaration will define a type `shape` that can be used to
refer to such shapes, and two functions, `circle` and `rectangle`,
which can be used to construct values of the type (taking arguments of
the specified types). So `circle({x: 0f, y: 0f}, 10f)` is the way to
create a new circle.

Enum variants do not have to have parameters. This, for example, is
equivalent to a C enum:

    enum direction {
        north;
        east;
        south;
        west;
    }

This will define `north`, `east`, `south`, and `west` as constants,
all of which have type `direction`.

<a name="single_variant_enum"></a>

There is a special case for enums with a single variant. These are
used to define new types in such a way that the new name is not just a
synonym for an existing type, but its own distinct type. If you say:

    enum gizmo_id = int;

That is a shorthand for this:

    enum gizmo_id { gizmo_id(int); }

Enum types like this can have their content extracted with the
dereference (`*`) unary operator:

    # enum gizmo_id = int;
    let my_gizmo_id = gizmo_id(10);
    let id_int: int = *my_gizmo_id;

## Enum patterns

For enum types with multiple variants, destructuring is the only way to
get at their contents. All variant constructors can be used as
patterns, as in this definition of `area`:

    # type point = {x: float, y: float};
    # enum shape { circle(point, float); rectangle(point, point); }
    fn area(sh: shape) -> float {
        alt sh {
            circle(_, size) { float::consts::pi * size * size }
            rectangle({x, y}, {x: x2, y: y2}) { (x2 - x) * (y2 - y) }
        }
    }

For variants without arguments, you have to write `variantname.` (with
a dot at the end) to match them in a pattern. This to prevent
ambiguity between matching a variant name and binding a new variable.

    # type point = {x: float, y: float};
    # enum direction { north; east; south; west; }
    fn point_from_direction(dir: direction) -> point {
        alt dir {
            north. { {x:  0f, y:  1f} }
            east.  { {x:  1f, y:  0f} }
            south. { {x:  0f, y: -1f} }
            west.  { {x: -1f, y:  0f} }
        }
    }

## Tuples

Tuples in Rust behave exactly like records, except that their fields
do not have names (and can thus not be accessed with dot notation).
Tuples can have any arity except for 0 or 1 (though you may see nil,
`()`, as the empty tuple if you like).

    let mytup: (int, int, float) = (10, 20, 30.0);
    alt mytup {
      (a, b, c) { log(info, a + b + (c as int)); }
    }

## Pointers

In contrast to a lot of modern languages, record and enum types in
Rust are not represented as pointers to allocated memory. They are,
like in C and C++, represented directly. This means that if you `let x
= {x: 1f, y: 1f};`, you are creating a record on the stack. If you
then copy it into a data structure, the whole record is copied, not
just a pointer.

For small records like `point`, this is usually more efficient than
allocating memory and going through a pointer. But for big records, or
records with mutable fields, it can be useful to have a single copy on
the heap, and refer to that through a pointer.

Rust supports several types of pointers. The simplest is the unsafe
pointer, written `*TYPE`, which is a completely unchecked pointer
type only used in unsafe code (and thus, in typical Rust code, very
rarely). The safe pointer types are `@TYPE` for shared,
reference-counted boxes, and `~TYPE`, for uniquely-owned pointers.

All pointer types can be dereferenced with the `*` unary operator.

### Shared boxes

<a name="shared-box"></a>

Shared boxes are pointers to heap-allocated, reference counted memory.
A cycle collector ensures that circular references do not result in
memory leaks.

Creating a shared box is done by simply applying the unary `@`
operator to an expression. The result of the expression will be boxed,
resulting in a box of the right type. For example:

    let x = @10; // New box, refcount of 1
    let y = x; // Copy the pointer, increase refcount
    // When x and y go out of scope, refcount goes to 0, box is freed

NOTE: We may in the future switch to garbage collection, rather than
reference counting, for shared boxes.

Shared boxes never cross task boundaries.

### Unique boxes

<a name="unique-box"></a>

In contrast to shared boxes, unique boxes are not reference counted.
Instead, it is statically guaranteed that only a single owner of the
box exists at any time.

    let x = ~10;
    let y <- x;

This is where the 'move' (`<-`) operator comes in. It is similar to
`=`, but it de-initializes its source. Thus, the unique box can move
from `x` to `y`, without violating the constraint that it only has a
single owner (if you used assignment instead of the move operator, the
box would, in principle, be copied).

Unique boxes, when they do not contain any shared boxes, can be sent
to other tasks. The sending task will give up ownership of the box,
and won't be able to access it afterwards. The receiving task will
become the sole owner of the box.

### Mutability

All pointer types have a mutable variant, written `@mutable TYPE` or
`~mutable TYPE`. Given such a pointer, you can write to its contents
by combining the dereference operator with a mutating action.

    fn increase_contents(pt: @mutable int) {
        *pt += 1;
    }

## Vectors

Rust vectors are always heap-allocated and unique. A value of type
`[TYPE]` is represented by a pointer to a section of heap memory
containing any number of `TYPE` values.

NOTE: This uniqueness is turning out to be quite awkward in practice,
and might change in the future.

Vector literals are enclosed in square brackets. Dereferencing is done
with square brackets (zero-based):

    let myvec = [true, false, true, false];
    if myvec[1] { std::io::println("boom"); }

By default, vectors are immutable—you can not replace their elements.
The type written as `[mutable TYPE]` is a vector with mutable
elements. Mutable vector literals are written `[mutable]` (empty) or
`[mutable 1, 2, 3]` (with elements).

The `+` operator means concatenation when applied to vector types.
Growing a vector in Rust is not as inefficient as it looks :

    let myvec = [], i = 0;
    while i < 100 {
        myvec += [i];
        i += 1;
    }

Because a vector is unique, replacing it with a longer one (which is
what `+= [i]` does) is indistinguishable from appending to it
in-place. Vector representations are optimized to grow
logarithmically, so the above code generates about the same amount of
copying and reallocation as `push` implementations in most other
languages.

## Strings

The `str` type in Rust is represented exactly the same way as a vector
of bytes (`[u8]`), except that it is guaranteed to have a trailing
null byte (for interoperability with C APIs).

This sequence of bytes is interpreted as an UTF-8 encoded sequence of
characters. This has the advantage that UTF-8 encoded I/O (which
should really be the default for modern systems) is very fast, and
that strings have, for most intents and purposes, a nicely compact
representation. It has the disadvantage that you only get
constant-time access by byte, not by character.

A lot of algorithms don't need constant-time indexed access (they
iterate over all characters, which `str::chars` helps with), and
for those that do, many don't need actual characters, and can operate
on bytes. For algorithms that do really need to index by character,
there's the option to convert your string to a character vector (using
`str::to_chars`).

Like vectors, strings are always unique. You can wrap them in a shared
box to share them. Unlike vectors, there is no mutable variant of
strings. They are always immutable.

## Resources

Resources are data types that have a destructor associated with them.

    # fn close_file_desc(x: int) {}
    resource file_desc(fd: int) {
        close_file_desc(fd);
    }

This defines a type `file_desc` and a constructor of the same name,
which takes an integer. Values of such a type can not be copied, and
when they are destroyed (by going out of scope, or, when boxed, when
their box is cleaned up), their body runs. In the example above, this
would cause the given file descriptor to be closed.

NOTE: We're considering alternative approaches for data types with
destructors. Resources might go away in the future.
