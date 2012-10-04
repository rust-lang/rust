% Rust Borrowed Pointers Tutorial

# Introduction

Borrowed pointers are one of the more flexible and powerful tools
available in Rust. A borrowed pointer can be used to point anywhere:
into the managed and exchange heaps, into the stack, and even into the
interior of another data structure. With regard to flexibility, it is
comparable to a C pointer or C++ reference. However, unlike C and C++,
the Rust compiler includes special checks that ensure that borrowed
pointers are being used safely. Another advantage of borrowed pointers
is that they are invisible to the garbage collector, so working with
borrowed pointers helps keep things efficient.

Despite the fact that they are completely safe, at runtime, a borrowed
pointer is “just a pointer”. They introduce zero overhead. All safety
checks are done at compilation time.

Although borrowed pointers have rather elaborate theoretical
underpinnings (region pointers), the core concepts will be familiar to
anyone who worked with C or C++. Therefore, the best way to explain
how they are used—and their limitations—is probably just to work
through several examples.

# By example

Borrowed pointers are called borrowed because they are only valid for
a limit duration. Borrowed pointers never claim any kind of ownership
over the data that they point at: instead, they are used for cases
where you like to make use of data for a short time.

As an example, consider a simple struct type `Point`:

~~~
struct Point {x: float, y: float}
~~~

We can use this simple definition to allocate points in many ways. For
example, in this code, each of these three local variables contains a
point, but allocated in a different place:

~~~
# struct Point {x: float, y: float}
let on_the_stack :  Point =  Point {x: 3.0, y: 4.0};
let shared_box   : @Point = @Point {x: 5.0, y: 1.0};
let unique_box   : ~Point = ~Point {x: 7.0, y: 9.0};
~~~

Suppose we wanted to write a procedure that computed the distance
between any two points, no matter where they were stored. For example,
we might like to compute the distance between `on_the_stack` and
`shared_box`, or between `shared_box` and `unique_box`. One option is
to define a function that takes two arguments of type point—that is,
it takes the points by value. But this will cause the points to be
copied when we call the function. For points, this is probably not so
bad, but often copies are expensive or, worse, if there are mutable
fields, they can change the semantics of your program. So we’d like to
define a function that takes the points by pointer. We can use
borrowed pointers to do this:

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

Here the `&` operator is used to take the address of the variable
`on_the_stack`; this is because `on_the_stack` has the type `Point`
(that is, a struct value) and we have to take its address to get a
value. We also call this _borrowing_ the local variable
`on_the_stack`, because we are created an alias: that is, another
route to the same data.

In the case of the boxes `shared_box` and `unique_box`, however, no
explicit action is necessary. The compiler will automatically convert
a box like `@Point` or `~Point` to a borrowed pointer like
`&Point`. This is another form of borrowing; in this case, the
contents of the shared/unique box is being lent out.

Whenever a value is borrowed, there are some limitations on what you
can do with the original. For example, if the contents of a variable
have been lent out, you cannot send that variable to another task, nor
will you be permitted to take actions that might cause the borrowed
value to be freed or to change its type (I’ll get into what kinds of
actions those are shortly). This rule should make intuitive sense: you
must wait for a borrowed value to be returned (that is, for the
borrowed pointer to go out of scope) before you can make full use of
it again.

# Other uses for the & operator

In the previous example, the value `on_the_stack` was defined like so:

~~~
# struct Point {x: float, y: float}
let on_the_stack: Point = Point {x: 3.0, y: 4.0};
~~~

This results in a by-value variable. As a consequence, we had to
explicitly take the address of `on_the_stack` to get a borrowed
pointer. Sometimes however it is more convenient to move the &
operator into the definition of `on_the_stack`:

~~~
# struct Point {x: float, y: float}
let on_the_stack2: &Point = &Point {x: 3.0, y: 4.0};
~~~

Applying `&` to an rvalue (non-assignable location) is just a convenient
shorthand for creating a temporary and taking its address:

~~~
# struct Point {x: float, y: float}
let tmp = Point {x: 3.0, y: 4.0};
let on_the_stack2 : &Point = &tmp;
~~~

# Taking the address of fields

As in C, the `&` operator is not limited to taking the address of
local variables. It can also be used to take the address of fields or
individual array elements. For example, consider this type definition
for `rectangle`:

~~~
struct Point {x: float, y: float} // as before
struct Size {w: float, h: float} // as before
struct Rectangle {origin: Point, size: Size}
~~~

Now again I can define rectangles in a few different ways:

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

In each case I can use the `&` operator to extact out individual
subcomponents. For example, I could write:

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
from the managed box and then compute the distance between them.

# Borrowing managed boxes and rooting

We’ve seen a few examples so far where heap boxes (both managed and
unique) are borrowed. Up till this point, we’ve glossed over issues of
safety. As stated in the introduction, at runtime a borrowed pointer
is simply a pointer, nothing more. Therefore, if we wish to avoid the
issues that C has with dangling pointers (and we do!), a compile-time
safety check is required.

The basis for the check is the notion of _lifetimes_. A lifetime is
basically a static approximation of the period in which the pointer is
valid: it always corresponds to some expression or block within the
program. Within that expression, the pointer can be used freely, but
if the pointer somehow leaks outside of that expression, the compiler
will report an error. We’ll be discussing lifetimes more in the
examples to come, and a more thorough introduction is also available.

When a borrowed pointer is created, the compiler must ensure that it
will remain valid for its entire lifetime. Sometimes this is
relatively easy, such as when taking the address of a local variable
or a field that is stored on the stack:

~~~
struct X { f: int }
fn example1() {
    let mut x = X { f: 3 };
    let y = &mut x.f;  // -+ L
    ...                //  |
}                      // -+
~~~

Here, the lifetime of the borrowed pointer is simply L, the remainder
of the function body. No extra work is required to ensure that `x.f`
will not be freed. This is true even if `x` is mutated.

The situation gets more complex when borrowing data that resides in
heap boxes:

~~~
# struct X { f: int }
fn example2() {
    let mut x = @X { f: 3 };
    let y = &x.f;      // -+ L
    ...                //  |
}                      // -+
~~~

In this example, the value `x` is in fact a heap box, and `y` is
therefore a pointer into that heap box. Again the lifetime of `y` will
be L, the remainder of the function body. But there is a crucial
difference: suppose `x` were reassigned during the lifetime L? If
we’re not careful, that could mean that the managed box would become
unrooted and therefore be subject to garbage collection

> ***Note:***In our current implementation, the garbage collector is
> implemented using reference counting and cycle detection.

For this reason, whenever the interior of a managed box stored in a
mutable location is borrowed, the compiler will insert a temporary
that ensures that the managed box remains live for the entire
lifetime. So, the above example would be compiled as:

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
borrow. Unfortunately, rooting does not work if the data being
borrowed is a unique box, as it is not possible to have two references
to a unique box.

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
and `x` is declared as mutable. However, the compiler can clearly see
that `x` is not assigned anywhere in the lifetime L of the variable
`y`. Therefore, it accepts the function, even though `x` is mutable
and in fact is mutated later in the function.

It may not be clear why we are so concerned about the variable which
was borrowed being mutated. The reason is that unique boxes are freed
_as soon as their owning reference is changed or goes out of
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

In fact, the compiler can apply this same kind of reasoning can be
applied to any memory which is _(uniquely) owned by the stack
frame_. So we could modify the previous example to introduce
additional unique pointers and structs, and the compiler will still be
able to detect possible mutations:

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
cause the pointer `y` to be invalidated.

Things get tricker when the unique box is not uniquely owned by the
stack frame (or when the compiler doesn’t know who the owner
is). Consider a program like this:

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

In this case, the owning reference to the value being borrowed is in
fact `x.f`. Moreover, `x.f` is both mutable and aliasable. Aliasable
means that it is possible that there are other pointers to that same
managed box, so even if the compiler were to prevent `x.f` from being
mutated, the field might still be changed through some alias of
`x`. Therefore, to be safe, the compiler only accepts pure actions
during the lifetime of `y`. We’ll have a final example on purity but
inn unique fields, as in the following example:

Besides ensuring purity, the only way to borrow the interior of a
unique found in aliasable memory is to ensure that it is stored within
unique fields, as in the following example:

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
it, one option is to use the swap operator to bring that unique box
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
    x.f <- v;          // Replace x.f
    ...
# return 0;
}
~~~

Of course, this has the side effect of modifying your managed box for
the duration of the borrow, so it only works when you know that you
won’t be accessing that same box for the duration of the loan.  Note
also that sometimes it is necessary to introduce additional blocks to
constrain the scope of the loan.  In this example, the borrowed
pointer `y` would still be in scope when you moved the value `v` back
into `x.f`, and hence moving `v` would be considered illegal.  You
cannot move values if they are outstanding loans which are still
valid.  By introducing the block, the scope of `y` is restricted and so
the move is legal.

# Borrowing and enums

The previous example showed that borrowing unique boxes found in
aliasable, mutable memory is not permitted, so as to prevent pointers
into freed memory. There is one other case where the compiler must be
very careful to ensure that pointers remain valid: pointers into the
interior of an enum.

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

Now I might write a function to compute the area of a shape. This
function takes a borrowed pointer to a shape to avoid the need of
copying them.

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

The first case matches against circles. Here the radius is extracted
from the shape variant and used to compute the area of the circle
(Like any up-to-date engineer, we use the [tau circle constant][tau]
and not that dreadfully outdated notion of pi).

[tau]: http://www.math.utah.edu/~palais/pi.html

The second match is more interesting. Here we match against a
rectangle and extract its size: but rather than copy the `size` struct,
we use a by-reference binding to create a pointer to it. In other
words, a pattern binding like `ref size` in fact creates a pointer of
type `&size` into the _interior of the enum_.

To make this more clear, let’s look at a diagram of how things are
laid out in memory in the case where `shape` points at a rectangle:

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
different type_! This is shown in the following diagram, depicting what
the state of memory would be if shape were overwritten with a circle:

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

As you can see, the `size` pointer would not be pointing at a `float` and
not a struct. This is not good.

So, in fact, for every `ref` binding, the compiler will impose the
same rules as the ones we saw for borrowing the interior of a unique
box: it must be able to guarantee that the enum will not be
overwritten for the duration of the borrow.  In fact, the example I
gave earlier would be considered safe. This is because the shape
pointer has type `&Shape`, which means “borrowed pointer to immutable
memory containing a shape”. If however the type of that pointer were
`&const Shape` or `&mut Shape`, then the ref binding would not be
permitted. Just as with unique boxes, the compiler will permit ref
bindings into data owned by the stack frame even if it is mutable, but
otherwise it requires that the data reside in immutable memory.

> ***Note:*** Right now, pattern bindings not explicitly annotated
> with `ref` or `copy` use a special mode of "implicit by reference".
> This is changing as soon as we finish updating all the existing code
> in the compiler that relies on the current settings.

# Returning borrowed pointers

So far, all of the examples we’ve looked at use borrowed pointers in a
“downward” direction. That is, the borrowed pointer is created and
then used during the method or code block which created it. It is also
possible to return borrowed pointers to the caller, but as we'll see
this requires some explicit annotation.

For example, we could write a subroutine like this:

~~~
struct Point {x: float, y: float}
fn get_x(p: &r/Point) -> &r/float { &p.x }
~~~

Here, the function `get_x()` returns a pointer into the structure it was
given. The type of the parameter (`&r/Point`) and return type (`&r/float`) both
make use of a new syntactic form that we have not seen so far.  Here the identifier `r`
serves as an explicit name for the lifetime of the pointer.  So in effect
this function is declaring that it takes in a pointer with lifetime `r` and returns
a pointer with that same lifetime.

In general, it is only possible to return borrowed pointers if they
are derived from a borrowed pointer which was given as input to the
procedure.  In that case, they will always have the same lifetime as
one of the parameters; named lifetimes are used to indicate which
parameter that is.

In the examples before, function parameter types did not include a
lifetime name.  In this case, the compiler simply creates a new,
anonymous name, meaning that the parameter is assumed to have a
distinct lifetime from all other parameters.

Named lifetimes that appear in function signatures are conceptually
the same as the other lifetimes we've seen before, but they are a bit
abstract: they don’t refer to a specific expression within `get_x()`,
but rather to some expression within the *caller of `get_x()`*.  The
lifetime `r` is actually a kind of *lifetime parameter*: it is defined
by the caller to `get_x()`, just as the value for the parameter `p` is
defined by that caller.

In any case, whatever the lifetime `r` is, the pointer produced by
`&p.x` always has the same lifetime as `p` itself, as a pointer to a
field of a struct is valid as long as the struct is valid. Therefore,
the compiler is satisfied with the function `get_x()`.

To drill in this point, let’s look at a variation on the example, this
time one which does not compile:

~~~ {.xfail-test}
struct Point {x: float, y: float}
fn get_x_sh(p: @Point) -> &float {
    &p.x // Error reported here
}
~~~

Here, the function `get_x_sh()` takes a managed box as input and
returns a borrowed pointer. As before, the lifetime of the borrowed
pointer that will be returned is a parameter (specified by the
caller). That means that effectively `get_x_sh()` is promising to
return a borrowed pointer that is valid for as long as the caller
would like: this is subtly different from the first example, which
promised to return a pointer that was valid for as long as the pointer
it was given.

Within `get_x_sh()`, we see the expression `&p.x` which takes the
address of a field of a managed box. This implies that the compiler
must guarantee that, so long as the resulting pointer is valid, the
managed box will not be reclaimed by the garbage collector. But recall
that `get_x_sh()` also promised to return a pointer that was valid for
as long as the caller wanted it to be. Clearly, `get_x_sh()` is not in
a position to make both of these guarantees; in fact, it cannot
guarantee that the pointer will remain valid at all once it returns,
as the parameter `p` may or may not be live in the caller. Therefore,
the compiler will report an error here.

In general, if you borrow a managed (or unique) box to create a
borrowed pointer, the pointer will only be valid within the function
and cannot be returned. This is why the typical way to return borrowed
pointers is to take borrowed pointers as input (the only other case in
which it can be legal to return a borrowed pointer is if the pointer
points at a static constant).

# Named lifetimes

Let's look at named lifetimes in more detail.  In effect, the use of
named lifetimes allows you to group parameters by lifetime.  For
example, consider this function:

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
value, and hence the return value of `shape()` will be assigned a
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

Here you can see the lifetime of shape is now being called `tmp`. The
parameters `a`, `b`, and the return value are all given the lifetime
`r`.  However, since the lifetime `tmp` is not returned, it would be shorter
to just omit the named lifetime for `shape` altogether:

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
that permits borrowing of any data, but only if the actions that occur
during the lifetime of the borrow are pure. Pure actions are those
which only modify data owned by the current stack frame. The compiler
can therefore permit arbitrary pointers into the heap, secure in the
knowledge that no pure action will ever cause them to become
invalidated (the compiler must still track data on the stack which is
borrowed and enforce those rules normally, of course).

Let’s revisit a previous example and show how purity can affect the
compiler’s result. Here is `example5a()`, which borrows the interior of
a unique box found in an aliasable, mutable location, only now we’ve
replaced the `...` with some specific code:

~~~
struct R { g: int }
struct S { mut f: ~R }
fn example5a(x: @S ...) -> int {
    let y = &x.f.g;   // Unsafe
    *y + 1        
}
~~~

The new code simply returns an incremented version of `y`. This clearly
doesn’t do mutate anything in the heap, so the compiler is satisfied.

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
only considers one function at a time (like most type checkers), and
so it does not know that `add_one()` only takes pure actions. We can
help the compiler by labeling `add_one()` as pure:

~~~
pure fn add_one(x: &int) -> int { *x + 1 }
~~~

With this change, the modified version of `example5a()` will again compile.

# Conclusion

So there you have it. A (relatively) brief tour of borrowed pointer
system. For more details, I refer to the (yet to be written) reference
document on borrowed pointers, which will explain the full notation
and give more examples.
