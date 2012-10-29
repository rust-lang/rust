% Rust Borrowed Pointers Tutorial

# Introduction

Borrowed pointers are one of the more flexible and powerful tools available in
Rust. A borrowed pointer can point anywhere: into the managed or exchange
heap, into the stack, and even into the interior of another data structure. A
borrowed pointer is as flexible as a C pointer or C++ reference. However,
unlike C and C++ compilers, the Rust compiler includes special static checks
that ensure that programs use borrowed pointers safely. Another advantage of
borrowed pointers is that they are invisible to the garbage collector, so
working with borrowed pointers helps reduce the overhead of automatic memory
management.

Despite their complete safety, a borrowed pointer's representation at runtime
is the same as that of an ordinary pointer in a C program. They introduce zero
overhead. The compiler does all safety checks at compile time.

Although borrowed pointers have rather elaborate theoretical
underpinnings (region pointers), the core concepts will be familiar to
anyone who has worked with C or C++. Therefore, the best way to explain
how they are used—and their limitations—is probably just to work
through several examples.

# By example

Borrowed pointers are called *borrowed* because they are only valid for
a limited duration. Borrowed pointers never claim any kind of ownership
over the data that they point to: instead, they are used for cases
where you would like to use data for a short time.

As an example, consider a simple struct type `Point`:

~~~
struct Point {x: float, y: float}
~~~

We can use this simple definition to allocate points in many different ways. For
example, in this code, each of these three local variables contains a
point, but allocated in a different place:

~~~
# struct Point {x: float, y: float}
let on_the_stack :  Point =  Point {x: 3.0, y: 4.0};
let shared_box   : @Point = @Point {x: 5.0, y: 1.0};
let unique_box   : ~Point = ~Point {x: 7.0, y: 9.0};
~~~

Suppose we wanted to write a procedure that computed the distance between any
two points, no matter where they were stored. For example, we might like to
compute the distance between `on_the_stack` and `shared_box`, or between
`shared_box` and `unique_box`. One option is to define a function that takes
two arguments of type `Point`—that is, it takes the points by value. But we
define it this way, calling the function will cause the points to be
copied. For points, this is probably not so bad, but often copies are
expensive. Worse, if the data type contains mutable fields, copying can change
the semantics of your program in unexpected ways. So we'd like to define a
function that takes the points by pointer. We can use borrowed pointers to do
this:

~~~
# struct Point {x: float, y: float}
# fn sqrt(f: float) -> float { 0f }
fn compute_distance(p1: &Point, p2: &Point) -> float {
    let x_d = p1.x - p2.x;
    let y_d = p1.y - p2.y;
    sqrt(x_d * x_d + y_d * y_d)
}
~~~

Now we can call `compute_distance()` in various ways:

~~~
# struct Point {x: float, y: float}
# let on_the_stack :  Point =  Point{x: 3.0, y: 4.0};
# let shared_box   : @Point = @Point{x: 5.0, y: 1.0};
# let unique_box   : ~Point = ~Point{x: 7.0, y: 9.0};
# fn compute_distance(p1: &Point, p2: &Point) -> float { 0f }
compute_distance(&on_the_stack, shared_box);
compute_distance(shared_box, unique_box);
~~~

Here, the `&` operator takes the address of the variable
`on_the_stack`; this is because `on_the_stack` has the type `Point`
(that is, a struct value) and we have to take its address to get a
value. We also call this _borrowing_ the local variable
`on_the_stack`, because we have created an alias: that is, another
name for the same data.

In contrast, we can pass the boxes `shared_box` and `unique_box` to
`compute_distance` directly. The compiler automatically converts a box like
`@Point` or `~Point` to a borrowed pointer like `&Point`. This is another form
of borrowing: in this case, the caller lends the contents of the shared or
unique box to the callee.

Whenever a caller lends data to a callee, there are some limitations on what
the caller can do with the original. For example, if the contents of a
variable have been lent out, you cannot send that variable to another task. In
addition, the compiler will reject any code that might cause the borrowed
value to be freed or overwrite its component fields with values of different
types (I'll get into what kinds of actions those are shortly). This rule
should make intuitive sense: you must wait for a borrower to return the value
that you lent it (that is, wait for the borrowed pointer to go out of scope)
before you can make full use of it again.

# Other uses for the & operator

In the previous example, the value `on_the_stack` was defined like so:

~~~
# struct Point {x: float, y: float}
let on_the_stack: Point = Point {x: 3.0, y: 4.0};
~~~

This declaration means that code can only pass `Point` by value to other
functions. As a consequence, we had to explicitly take the address of
`on_the_stack` to get a borrowed pointer. Sometimes however it is more
convenient to move the & operator into the definition of `on_the_stack`:

~~~
# struct Point {x: float, y: float}
let on_the_stack2: &Point = &Point {x: 3.0, y: 4.0};
~~~

Applying `&` to an rvalue (non-assignable location) is just a convenient
shorthand for creating a temporary and taking its address. A more verbose
way to write the same code is:

~~~
# struct Point {x: float, y: float}
let tmp = Point {x: 3.0, y: 4.0};
let on_the_stack2 : &Point = &tmp;
~~~

# Taking the address of fields

As in C, the `&` operator is not limited to taking the address of
local variables. It can also take the address of fields or
individual array elements. For example, consider this type definition
for `rectangle`:

~~~
struct Point {x: float, y: float} // as before
struct Size {w: float, h: float} // as before
struct Rectangle {origin: Point, size: Size}
~~~

Now, as before, we can define rectangles in a few different ways:

~~~
# struct Point {x: float, y: float}
# struct Size {w: float, h: float} // as before
# struct Rectangle {origin: Point, size: Size}
let rect_stack   = &Rectangle {origin: Point {x: 1f, y: 2f},
                               size: Size {w: 3f, h: 4f}};
let rect_managed = @Rectangle {origin: Point {x: 3f, y: 4f},
                               size: Size {w: 3f, h: 4f}};
let rect_unique  = ~Rectangle {origin: Point {x: 5f, y: 6f},
                               size: Size {w: 3f, h: 4f}};
~~~

In each case, we can extract out individual subcomponents with the `&`
operator. For example, I could write:

~~~
# struct Point {x: float, y: float} // as before
# struct Size {w: float, h: float} // as before
# struct Rectangle {origin: Point, size: Size}
# let rect_stack  = &{origin: Point {x: 1f, y: 2f}, size: Size {w: 3f, h: 4f}};
# let rect_managed = @{origin: Point {x: 3f, y: 4f}, size: Size {w: 3f, h: 4f}};
# let rect_unique = ~{origin: Point {x: 5f, y: 6f}, size: Size {w: 3f, h: 4f}};
# fn compute_distance(p1: &Point, p2: &Point) -> float { 0f }
compute_distance(&rect_stack.origin, &rect_managed.origin);
~~~

which would borrow the field `origin` from the rectangle on the stack
as well as from the managed box, and then compute the distance between them.

# Borrowing managed boxes and rooting

We’ve seen a few examples so far of borrowing heap boxes, both managed
and unique. Up till this point, we’ve glossed over issues of
safety. As stated in the introduction, at runtime a borrowed pointer
is simply a pointer, nothing more. Therefore, avoiding C's problems
with dangling pointers requires a compile-time safety check.

The basis for the check is the notion of _lifetimes_. A lifetime is a
static approximation of the span of execution during which the pointer
is valid: it always corresponds to some expression or block within the
program. Code inside that expression can use the pointer without
restrictions. But if the pointer escapes from that expression (for
example, if the expression contains an assignment expression that
assigns the pointer to a mutable field of a data structure with a
broader scope than the pointer itself), the compiler reports an
error. We'll be discussing lifetimes more in the examples to come, and
a more thorough introduction is also available.

When the `&` operator creates a borrowed pointer, the compiler must
ensure that the pointer remains valid for its entire
lifetime. Sometimes this is relatively easy, such as when taking the
address of a local variable or a field that is stored on the stack:

~~~
struct X { f: int }
fn example1() {
    let mut x = X { f: 3 };
    let y = &mut x.f;  // -+ L
    ...                //  |
}                      // -+
~~~

Here, the lifetime of the borrowed pointer `y` is simply L, the
remainder of the function body. The compiler need not do any other
work to prove that code will not free `x.f`. This is true even if the
code mutates `x`.

The situation gets more complex when borrowing data inside heap boxes:

~~~
# struct X { f: int }
fn example2() {
    let mut x = @X { f: 3 };
    let y = &x.f;      // -+ L
    ...                //  |
}                      // -+
~~~

In this example, the value `x` is a heap box, and `y` is therefore a
pointer into that heap box. Again the lifetime of `y` is L, the
remainder of the function body. But there is a crucial difference:
suppose `x` were to be reassigned during the lifetime L? If the
compiler isn't careful, the managed box could become *unrooted*, and
would therefore be subject to garbage collection. A heap box that is
unrooted is one such that no pointer values in the heap point to
it. It would violate memory safety for the box that was originally
assigned to `x` to be garbage-collected, since a non-heap
pointer---`y`---still points into it.

> ***Note:*** Our current implementation implements the garbage collector
> using reference counting and cycle detection.

For this reason, whenever an `&` expression borrows the interior of a
managed box stored in a mutable location, the compiler inserts a
temporary that ensures that the managed box remains live for the
entire lifetime. So, the above example would be compiled as if it were
written

~~~
# struct X { f: int }
fn example2() {
    let mut x = @X {f: 3};
    let x1 = x;
    let y = &x1.f;     // -+ L
    ...                //  |
}                      // -+
~~~

Now if `x` is reassigned, the pointer `y` will still remain valid. This
process is called *rooting*.

# Borrowing unique boxes

The previous example demonstrated *rooting*, the process by which the
compiler ensures that managed boxes remain live for the duration of a
borrow. Unfortunately, rooting does not work for borrows of unique
boxes, because it is not possible to have two references to a unique
box.

For unique boxes, therefore, the compiler will only allow a borrow *if
the compiler can guarantee that the unique box will not be reassigned
or moved for the lifetime of the pointer*. This does not necessarily
mean that the unique box is stored in immutable memory. For example,
the following function is legal:

~~~
# fn some_condition() -> bool { true }
fn example3() -> int {
    let mut x = ~{f: 3};
    if some_condition() {
        let y = &x.f;      // -+ L
        return *y;         //  |
    }                      // -+
    x = ~{f: 4};
    ...
# return 0;
}
~~~

Here, as before, the interior of the variable `x` is being borrowed
and `x` is declared as mutable. However, the compiler can prove that
`x` is not assigned anywhere in the lifetime L of the variable
`y`. Therefore, it accepts the function, even though `x` is mutable
and in fact is mutated later in the function.

It may not be clear why we are so concerned about mutating a borrowed
variable. The reason is that the runtime system frees any unique box
_as soon as its owning reference changes or goes out of
scope_. Therefore, a program like this is illegal (and would be
rejected by the compiler):

~~~ {.xfail-test}
fn example3() -> int {
    let mut x = ~X {f: 3};
    let y = &x.f;
    x = ~{f: 4};  // Error reported here.
    *y
}
~~~

To make this clearer, consider this diagram showing the state of
memory immediately before the re-assignment of `x`:

~~~ {.notrust}
    Stack               Exchange Heap

  x +----------+
    | ~{f:int} | ----+
  y +----------+     |
    | &int     | ----+
    +----------+     |    +---------+
                     +--> |  f: 3   |
                          +---------+
~~~

Once the reassignment occurs, the memory will look like this:

~~~ {.notrust}
    Stack               Exchange Heap

  x +----------+          +---------+
    | ~{f:int} | -------> |  f: 4   |
  y +----------+          +---------+
    | &int     | ----+
    +----------+     |    +---------+
                     +--> | (freed) |
                          +---------+
~~~

Here you can see that the variable `y` still points at the old box,
which has been freed.

In fact, the compiler can apply the same kind of reasoning to any
memory that is _(uniquely) owned by the stack frame_. So we could
modify the previous example to introduce additional unique pointers
and structs, and the compiler will still be able to detect possible
mutations:

~~~ {.xfail-test}
fn example3() -> int {
    struct R { g: int }
    struct S { mut f: ~R }

    let mut x = ~S {mut f: ~R {g: 3}};
    let y = &x.f.g;
    x = ~S {mut f: ~R {g: 4}}; // Error reported here.
    x.f = ~R {g: 5};           // Error reported here.
    *y
}
~~~

In this case, two errors are reported, one when the variable `x` is
modified and another when `x.f` is modified. Either modification would
invalidate the pointer `y`.

Things get trickier when the unique box is not uniquely owned by the
stack frame, or when there is no way for the compiler to determine the
box's owner. Consider a program like this:

~~~
struct R { g: int }
struct S { mut f: ~R }
fn example5a(x: @S ...) -> int {
    let y = &x.f.g;   // Error reported here.
    ...
#   return 0;
}
~~~

Here the heap looks something like:

~~~ {.notrust}
     Stack            Managed Heap       Exchange Heap

  x +------+        +-------------+       +------+
    | @... | ---->  | mut f: ~... | --+-> | g: 3 |
  y +------+        +-------------+   |   +------+
    | &int | -------------------------+
    +------+
~~~

In this case, the owning reference to the value being borrowed is
`x.f`. Moreover, `x.f` is both mutable and *aliasable*. Aliasable
means that there may be other pointers to that same managed box, so
even if the compiler were to prove an absence of mutations to `x.f`,
code could mutate `x.f` indirectly by changing an alias of
`x`. Therefore, to be safe, the compiler only accepts *pure* actions
during the lifetime of `y`. We define what "pure" means in the section
on [purity](#purity).

Besides ensuring purity, the only way to borrow the interior of a
unique found in aliasable memory is to ensure that the borrowed field
itself is also unique, as in the following example:

~~~
struct R { g: int }
struct S { f: ~R }
fn example5b(x: @S) -> int {
    let y = &x.f.g;
    ...
# return 0;
}
~~~

Here, the field `f` is not declared as mutable. But that is enough for
the compiler to know that, even if aliases to `x` exist, the field `f`
cannot be changed and hence the unique box `g` will remain valid.

If you do have a unique box in a mutable field, and you wish to borrow
it, one option is to use the swap operator to move that unique box
onto your stack:

~~~
struct R { g: int }
struct S { mut f: ~R }
fn example5c(x: @S) -> int {
    let mut v = ~R {g: 0};
    v <-> x.f;         // Swap v and x.f
    { // Block constrains the scope of `y`:
        let y = &v.g;
        ...
    }
    x.f = move v;          // Replace x.f
    ...
# return 0;
}
~~~

Of course, this has the side effect of modifying your managed box for
the duration of the borrow, so it only works when you know that you
won't be accessing that same box for the duration of the loan. Also,
it is sometimes necessary to introduce additional blocks to constrain
the scope of the loan.  In this example, the borrowed pointer `y`
would still be in scope when you moved the value `v` back into `x.f`,
and hence moving `v` would be considered illegal.  You cannot move
values if they are the targets of valid outstanding loans. Introducing
the block restricts the scope of `y`, making the move legal.

# Borrowing and enums

The previous example showed that the type system forbids any borrowing
of unique boxes found in aliasable, mutable memory. This restriction
prevents pointers from pointing into freed memory. There is one other
case where the compiler must be very careful to ensure that pointers
remain valid: pointers into the interior of an `enum`.

As an example, let’s look at the following `shape` type that can
represent both rectangles and circles:

~~~
struct Point {x: float, y: float}; // as before
struct Size {w: float, h: float}; // as before
enum Shape {
    Circle(Point, float),   // origin, radius
    Rectangle(Point, Size)  // upper-left, dimensions
}
~~~

Now we might write a function to compute the area of a shape. This
function takes a borrowed pointer to a shape, to avoid the need for
copying.

~~~
# struct Point {x: float, y: float}; // as before
# struct Size {w: float, h: float}; // as before
# enum Shape {
#     Circle(Point, float),   // origin, radius
#     Rectangle(Point, Size)  // upper-left, dimensions
# }
# const tau: float = 6.28f;
fn compute_area(shape: &Shape) -> float {
    match *shape {
        Circle(_, radius) => 0.5 * tau * radius * radius,
        Rectangle(_, ref size) => size.w * size.h
    }
}
~~~

The first case matches against circles. Here, the pattern extracts the
radius from the shape variant and the action uses it to compute the
area of the circle. (Like any up-to-date engineer, we use the [tau
circle constant][tau] and not that dreadfully outdated notion of pi).

[tau]: http://www.math.utah.edu/~palais/pi.html

The second match is more interesting. Here we match against a
rectangle and extract its size: but rather than copy the `size`
struct, we use a by-reference binding to create a pointer to it. In
other words, a pattern binding like `ref size` binds the name `size`
to a pointer of type `&size` into the _interior of the enum_.

To make this more clear, let's look at a diagram of memory layout in
the case where `shape` points at a rectangle:

~~~ {.notrust}
Stack             Memory

+-------+         +---------------+
| shape | ------> | rectangle(    |
+-------+         |   {x: float,  |
| size  | -+      |    y: float}, |
+-------+  +----> |   {w: float,  |
                  |    h: float}) |
                  +---------------+
~~~

Here you can see that rectangular shapes are composed of five words of
memory. The first is a tag indicating which variant this enum is
(`rectangle`, in this case). The next two words are the `x` and `y`
fields for the point and the remaining two are the `w` and `h` fields
for the size. The binding `size` is then a pointer into the inside of
the shape.

Perhaps you can see where the danger lies: if the shape were somehow
to be reassigned, perhaps to a circle, then although the memory used
to store that shape value would still be valid, _it would have a
different type_! The following diagram shows what memory would look
like if code overwrote `shape` with a circle:

~~~ {.notrust}
Stack             Memory

+-------+         +---------------+
| shape | ------> | circle(       |
+-------+         |   {x: float,  |
| size  | -+      |    y: float}, |
+-------+  +----> |   float)      |
                  |               |
                  +---------------+
~~~

As you can see, the `size` pointer would be pointing at a `float`
instead of a struct. This is not good: dereferencing the second field
of a `float` as if it were a struct with two fields would be a memory
safety violation.

So, in fact, for every `ref` binding, the compiler will impose the
same rules as the ones we saw for borrowing the interior of a unique
box: it must be able to guarantee that the `enum` will not be
overwritten for the duration of the borrow.  In fact, the compiler
would accept the example we gave earlier. The example is safe because
the shape pointer has type `&Shape`, which means "borrowed pointer to
immutable memory containing a `shape`". If, however, the type of that
pointer were `&const Shape` or `&mut Shape`, then the ref binding
would be ill-typed. Just as with unique boxes, the compiler will
permit `ref` bindings into data owned by the stack frame even if the
data are mutable, but otherwise it requires that the data reside in
immutable memory.

> ***Note:*** Right now, pattern bindings not explicitly annotated
> with `ref` or `copy` use a special mode of "implicit by reference".
> This is changing as soon as we finish updating all the existing code
> in the compiler that relies on the current settings.

# Returning borrowed pointers

So far, all of the examples we've looked at use borrowed pointers in a
“downward” direction. That is, a method or code block creates a
borrowed pointer, then uses it within the same scope. It is also
possible to return borrowed pointers as the result of a function, but
as we'll see, doing so requires some explicit annotation.

For example, we could write a subroutine like this:

~~~
struct Point {x: float, y: float}
fn get_x(p: &r/Point) -> &r/float { &p.x }
~~~

Here, the function `get_x()` returns a pointer into the structure it
was given. The type of the parameter (`&r/Point`) and return type
(`&r/float`) both use a new syntactic form that we have not seen so
far.  Here the identifier `r` names the lifetime of the pointer
explicitly. So in effect, this function declares that it takes a
pointer with lifetime `r` and returns a pointer with that same
lifetime.

In general, it is only possible to return borrowed pointers if they
are derived from a parameter to the procedure. In that case, the
pointer result will always have the same lifetime as one of the
parameters; named lifetimes indicate which parameter that
is.

In the previous examples, function parameter types did not include a
lifetime name. In those examples, the compiler simply creates a fresh
name for the lifetime automatically: that is, the lifetime name is
guaranteed to refer to a distinct lifetime from the lifetimes of all
other parameters.

Named lifetimes that appear in function signatures are conceptually
the same as the other lifetimes we've seen before, but they are a bit
abstract: they don’t refer to a specific expression within `get_x()`,
but rather to some expression within the *caller of `get_x()`*.  The
lifetime `r` is actually a kind of *lifetime parameter*: it is defined
by the caller to `get_x()`, just as the value for the parameter `p` is
defined by that caller.

In any case, whatever the lifetime of `r` is, the pointer produced by
`&p.x` always has the same lifetime as `p` itself: a pointer to a
field of a struct is valid as long as the struct is valid. Therefore,
the compiler accepts the function `get_x()`.

To emphasize this point, let’s look at a variation on the example, this
time one that does not compile:

~~~ {.xfail-test}
struct Point {x: float, y: float}
fn get_x_sh(p: @Point) -> &float {
    &p.x // Error reported here
}
~~~

Here, the function `get_x_sh()` takes a managed box as input and
returns a borrowed pointer. As before, the lifetime of the borrowed
pointer that will be returned is a parameter (specified by the
caller). That means that `get_x_sh()` promises to return a borrowed
pointer that is valid for as long as the caller would like: this is
subtly different from the first example, which promised to return a
pointer that was valid for as long as its pointer argument was valid.

Within `get_x_sh()`, we see the expression `&p.x` which takes the
address of a field of a managed box. The presence of this expression
implies that the compiler must guarantee that, so long as the
resulting pointer is valid, the managed box will not be reclaimed by
the garbage collector. But recall that `get_x_sh()` also promised to
return a pointer that was valid for as long as the caller wanted it to
be. Clearly, `get_x_sh()` is not in a position to make both of these
guarantees; in fact, it cannot guarantee that the pointer will remain
valid at all once it returns, as the parameter `p` may or may not be
live in the caller. Therefore, the compiler will report an error here.

In general, if you borrow a managed (or unique) box to create a
borrowed pointer, the pointer will only be valid within the function
and cannot be returned. This is why the typical way to return borrowed
pointers is to take borrowed pointers as input (the only other case in
which it can be legal to return a borrowed pointer is if the pointer
points at a static constant).

# Named lifetimes

Let's look at named lifetimes in more detail. Named lifetimes allow
for grouping of parameters by lifetime. For example, consider this
function:

~~~
# struct Point {x: float, y: float}; // as before
# struct Size {w: float, h: float}; // as before
# enum Shape {
#     Circle(Point, float),   // origin, radius
#     Rectangle(Point, Size)  // upper-left, dimensions
# }
# fn compute_area(shape: &Shape) -> float { 0f }
fn select<T>(shape: &r/Shape, threshold: float,
             a: &r/T, b: &r/T) -> &r/T {
    if compute_area(shape) > threshold {a} else {b}
}
~~~

This function takes three borrowed pointers and assigns each the same
lifetime `r`.  In practice, this means that, in the caller, the
lifetime `r` will be the *intersection of the lifetime of the three
region parameters*. This may be overly conservative, as in this
example:

~~~
# struct Point {x: float, y: float}; // as before
# struct Size {w: float, h: float}; // as before
# enum Shape {
#     Circle(Point, float),   // origin, radius
#     Rectangle(Point, Size)  // upper-left, dimensions
# }
# fn compute_area(shape: &Shape) -> float { 0f }
# fn select<T>(shape: &Shape, threshold: float,
#              a: &r/T, b: &r/T) -> &r/T {
#     if compute_area(shape) > threshold {a} else {b}
# }
                                                  // -+ r
fn select_based_on_unit_circle<T>(                //  |-+ B
    threshold: float, a: &r/T, b: &r/T) -> &r/T { //  | |
                                                  //  | |
    let shape = Circle(Point {x: 0., y: 0.}, 1.); //  | |
    select(&shape, threshold, a, b)               //  | |
}                                                 //  |-+
                                                  // -+
~~~

In this call to `select()`, the lifetime of the first parameter shape
is B, the function body. Both of the second two parameters `a` and `b`
share the same lifetime, `r`, which is a lifetime parameter of
`select_based_on_unit_circle()`. The caller will infer the
intersection of these two lifetimes as the lifetime of the returned
value, and hence the return value of `select()` will be assigned a
lifetime of B. This will in turn lead to a compilation error, because
`select_based_on_unit_circle()` is supposed to return a value with the
lifetime `r`.

To address this, we can modify the definition of `select()` to
distinguish the lifetime of the first parameter from the lifetime of
the latter two. After all, the first parameter is not being
returned. Here is how the new `select()` might look:

~~~
# struct Point {x: float, y: float}; // as before
# struct Size {w: float, h: float}; // as before
# enum Shape {
#     Circle(Point, float),   // origin, radius
#     Rectangle(Point, Size)  // upper-left, dimensions
# }
# fn compute_area(shape: &Shape) -> float { 0f }
fn select<T>(shape: &tmp/Shape, threshold: float,
             a: &r/T, b: &r/T) -> &r/T {
    if compute_area(shape) > threshold {a} else {b}
}
~~~

Here you can see that `shape`'s lifetime is now named `tmp`. The
parameters `a`, `b`, and the return value all have the lifetime `r`.
However, since the lifetime `tmp` is not returned, it would be more
concise to just omit the named lifetime for `shape` altogether:

~~~
# struct Point {x: float, y: float}; // as before
# struct Size {w: float, h: float}; // as before
# enum Shape {
#     Circle(Point, float),   // origin, radius
#     Rectangle(Point, Size)  // upper-left, dimensions
# }
# fn compute_area(shape: &Shape) -> float { 0f }
fn select<T>(shape: &Shape, threshold: float,
             a: &r/T, b: &r/T) -> &r/T {
    if compute_area(shape) > threshold {a} else {b}
}
~~~

This is equivalent to the previous definition.

# Purity

As mentioned before, the Rust compiler offers a kind of escape hatch
that permits borrowing of any data, as long as the actions that occur
during the lifetime of the borrow are pure. Pure actions are those
that only modify data owned by the current stack frame. The compiler
can therefore permit arbitrary pointers into the heap, secure in the
knowledge that no pure action will ever cause them to become
invalidated (the compiler must still track data on the stack which is
borrowed and enforce those rules normally, of course). A pure function
in Rust is referentially transparent: it returns the same results
given the same (observably equivalent) inputs. That is because while
pure functions are allowed to modify data, they may only modify
*stack-local* data, which cannot be observed outside the scope of the
function itself. (Using an `unsafe` block invalidates this guarantee.)

Let’s revisit a previous example and show how purity can affect
typechecking. Here is `example5a()`, which borrows the interior of a
unique box found in an aliasable, mutable location, only now we’ve
replaced the `...` with some specific code:

~~~
struct R { g: int }
struct S { mut f: ~R }
fn example5a(x: @S ...) -> int {
    let y = &x.f.g;   // Unsafe
    *y + 1        
}
~~~

The new code simply returns an incremented version of `y`. This code
clearly doesn't mutate the heap, so the compiler is satisfied.

But suppose we wanted to pull the increment code into a helper, like
this:

~~~
fn add_one(x: &int) -> int { *x + 1 }
~~~

We can now update `example5a()` to use `add_one()`:

~~~
# struct R { g: int }
# struct S { mut f: ~R }
# pure fn add_one(x: &int) -> int { *x + 1 }
fn example5a(x: @S ...) -> int {
    let y = &x.f.g;
    add_one(y)        // Error reported here
}
~~~

But now the compiler will report an error again. The reason is that it
only considers one function at a time (like most typecheckers), and
so it does not know that `add_one()` consists of pure code. We can
help the compiler by labeling `add_one()` as pure:

~~~
pure fn add_one(x: &int) -> int { *x + 1 }
~~~

With this change, the modified version of `example5a()` will again compile.

# Conclusion

So there you have it: a (relatively) brief tour of the borrowed pointer
system. For more details, we refer to the (yet to be written) reference
document on borrowed pointers, which will explain the full notation
and give more examples.
