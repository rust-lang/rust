% The Rust References and Lifetimes Guide

# Introduction

References are one of the more flexible and powerful tools available in
Rust. They can point anywhere: into the heap, stack, and even into the
interior of another data structure. A reference is as flexible as a C pointer
or C++ reference.

Unlike C and C++ compilers, the Rust compiler includes special static
checks that ensure that programs use references safely.

Despite their complete safety, a reference's representation at runtime
is the same as that of an ordinary pointer in a C program. They introduce zero
overhead. The compiler does all safety checks at compile time.

Although references have rather elaborate theoretical underpinnings usually
introduced as (e.g. region pointers), the core concepts will be familiar to
anyone who has worked with C or C++. The best way to explain how they are
used—and their limitations—is probably just to work through several examples.

# By example

References, sometimes known as *borrowed pointers*, are only valid for
a limited duration. References never claim any kind of ownership
over the data that they point to. Instead, they are used for cases
where you would like to use data for a short time.

Consider a simple struct type `Point`:

~~~
struct Point {x: f64, y: f64}
~~~

We can use this simple definition to allocate points in many different ways. For
example, in this code, each of these local variables contains a point,
but allocated in a different place:

~~~
# struct Point {x: f64, y: f64}
let on_the_stack : Point      =     Point {x: 3.0, y: 4.0};
let on_the_heap  : Box<Point> = box Point {x: 7.0, y: 9.0};
~~~

Suppose we wanted to write a procedure that computed the distance between any
two points, no matter where they were stored. One option is to define a function
that takes two arguments of type `Point`—that is, it takes the points by value.
But if we define it this way, calling the function will cause the points to be
copied. For points, this is probably not so bad, but often copies are
expensive. So we'd like to define a function that takes the points just as
a reference.

~~~
# struct Point {x: f64, y: f64}
# fn sqrt(f: f64) -> f64 { 0.0 }
fn compute_distance(p1: &Point, p2: &Point) -> f64 {
    let x_d = p1.x - p2.x;
    let y_d = p1.y - p2.y;
    sqrt(x_d * x_d + y_d * y_d)
}
~~~

Now we can call `compute_distance()`:

~~~
# struct Point {x: f64, y: f64}
# let on_the_stack :     Point  =     Point{x: 3.0, y: 4.0};
# let on_the_heap  : Box<Point> = box Point{x: 7.0, y: 9.0};
# fn compute_distance(p1: &Point, p2: &Point) -> f64 { 0.0 }
compute_distance(&on_the_stack, on_the_heap);
~~~

Here, the `&` operator takes the address of the variable
`on_the_stack`; this is because `on_the_stack` has the type `Point`
(that is, a struct value) and we have to take its address to get a
value. We also call this _borrowing_ the local variable
`on_the_stack`, because we have created an alias: that is, another
name for the same data.

In the case of `on_the_heap`, however, no explicit action is necessary. 
The compiler will automatically convert a box box point to a reference like &point. 
This is another form of borrowing; in this case, the contents of the owned box 
are being lent out.

Whenever a caller lends data to a callee, there are some limitations on what
the caller can do with the original. For example, if the contents of a
variable have been lent out, you cannot send that variable to another task. In
addition, the compiler will reject any code that might cause the borrowed
value to be freed or overwrite its component fields with values of different
types (I'll get into what kinds of actions those are shortly). This rule
should make intuitive sense: you must wait for a borrower to return the value
that you lent it (that is, wait for the reference to go out of scope)
before you can make full use of it again.

# Other uses for the & operator

In the previous example, the value `on_the_stack` was defined like so:

~~~
# struct Point {x: f64, y: f64}
let on_the_stack: Point = Point {x: 3.0, y: 4.0};
~~~

This declaration means that code can only pass `Point` by value to other
functions. As a consequence, we had to explicitly take the address of
`on_the_stack` to get a reference. Sometimes however it is more
convenient to move the & operator into the definition of `on_the_stack`:

~~~
# struct Point {x: f64, y: f64}
let on_the_stack2: &Point = &Point {x: 3.0, y: 4.0};
~~~

Applying `&` to an rvalue (non-assignable location) is just a convenient
shorthand for creating a temporary and taking its address. A more verbose
way to write the same code is:

~~~
# struct Point {x: f64, y: f64}
let tmp = Point {x: 3.0, y: 4.0};
let on_the_stack2 : &Point = &tmp;
~~~

# Taking the address of fields

The `&` operator is not limited to taking the address of
local variables. It can also take the address of fields or
individual array elements. For example, consider this type definition
for `Rectangle`:

~~~
struct Point {x: f64, y: f64} // as before
struct Size {w: f64, h: f64} // as before
struct Rectangle {origin: Point, size: Size}
~~~

Now, as before, we can define rectangles in a few different ways:

~~~
# struct Point {x: f64, y: f64}
# struct Size {w: f64, h: f64} // as before
# struct Rectangle {origin: Point, size: Size}
let rect_stack   =    &Rectangle {origin: Point {x: 1.0, y: 2.0},
                                  size: Size {w: 3.0, h: 4.0}};
let rect_heap    = box Rectangle {origin: Point {x: 5.0, y: 6.0},
                                  size: Size {w: 3.0, h: 4.0}};
~~~

In each case, we can extract out individual subcomponents with the `&`
operator. For example, I could write:

~~~
# struct Point {x: f64, y: f64} // as before
# struct Size {w: f64, h: f64} // as before
# struct Rectangle {origin: Point, size: Size}
# let rect_stack  = &Rectangle {origin: Point {x: 1.0, y: 2.0}, size: Size {w: 3.0, h: 4.0}};
# let rect_heap   = box Rectangle {origin: Point {x: 5.0, y: 6.0}, size: Size {w: 3.0, h: 4.0}};
# fn compute_distance(p1: &Point, p2: &Point) -> f64 { 0.0 }
compute_distance(&rect_stack.origin, &rect_heap.origin);
~~~

which would borrow the field `origin` from the rectangle on the stack
as well as from the owned box, and then compute the distance between them.

# Lifetimes

We’ve seen a few examples of borrowing data. To this point, we’ve glossed
over issues of safety. As stated in the introduction, at runtime a reference
is simply a pointer, nothing more. Therefore, avoiding C's problems with
dangling pointers requires a compile-time safety check.

The basis for the check is the notion of _lifetimes_. A lifetime is a
static approximation of the span of execution during which the pointer
is valid: it always corresponds to some expression or block within the
program.

The compiler will only allow a borrow *if it can guarantee that the data will
not be reassigned or moved for the lifetime of the pointer*. This does not
necessarily mean that the data is stored in immutable memory. For example,
the following function is legal:

~~~
# fn some_condition() -> bool { true }
# struct Foo { f: int }
fn example3() -> int {
    let mut x = box Foo {f: 3};
    if some_condition() {
        let y = &x.f;      // -+ L
        return *y;         //  |
    }                      // -+
    x = box Foo {f: 4};
    // ...
# return 0;
}
~~~

Here, the interior of the variable `x` is being borrowed
and `x` is declared as mutable. However, the compiler can prove that
`x` is not assigned anywhere in the lifetime L of the variable
`y`. Therefore, it accepts the function, even though `x` is mutable
and in fact is mutated later in the function.

It may not be clear why we are so concerned about mutating a borrowed
variable. The reason is that the runtime system frees any box
_as soon as its owning reference changes or goes out of
scope_. Therefore, a program like this is illegal (and would be
rejected by the compiler):

~~~ {.ignore}
fn example3() -> int {
    let mut x = box X {f: 3};
    let y = &x.f;
    x = box X {f: 4};  // Error reported here.
    *y
}
~~~

To make this clearer, consider this diagram showing the state of
memory immediately before the re-assignment of `x`:

~~~ {.notrust}
    Stack               Exchange Heap

  x +-------------+
    | box {f:int} | ----+
  y +-------------+     |
    | &int        | ----+
    +-------------+     |    +---------+
                        +--> |  f: 3   |
                             +---------+
~~~

Once the reassignment occurs, the memory will look like this:

~~~ {.notrust}
    Stack               Exchange Heap

  x +-------------+          +---------+
    | box {f:int} | -------> |  f: 4   |
  y +-------------+          +---------+
    | &int        | ----+
    +-------------+     |    +---------+
                        +--> | (freed) |
                             +---------+
~~~

Here you can see that the variable `y` still points at the old `f`
property of Foo, which has been freed.

In fact, the compiler can apply the same kind of reasoning to any
memory that is (uniquely) owned by the stack frame. So we could
modify the previous example to introduce additional owned pointers
and structs, and the compiler will still be able to detect possible
mutations. This time, we'll use an analogy to illustrate the concept.

~~~ {.ignore}
fn example3() -> int {
    struct House { owner: Box<Person> }
    struct Person { age: int }

    let mut house = box House {
        owner: box Person {age: 30}
    };

    let owner_age = &house.owner.age;
    house = box House {owner: box Person {age: 40}};  // Error reported here.
    house.owner = box Person {age: 50};               // Error reported here.
    *owner_age
}
~~~

In this case, two errors are reported, one when the variable `house` is
modified and another when `house.owner` is modified. Either modification would
invalidate the pointer `owner_age`.

# Borrowing and enums

The previous example showed that the type system forbids any borrowing
of owned boxes found in aliasable, mutable memory. This restriction
prevents pointers from pointing into freed memory. There is one other
case where the compiler must be very careful to ensure that pointers
remain valid: pointers into the interior of an `enum`.

Let’s look at the following `shape` type that can represent both rectangles
and circles:

~~~
struct Point {x: f64, y: f64}; // as before
struct Size {w: f64, h: f64}; // as before
enum Shape {
    Circle(Point, f64),   // origin, radius
    Rectangle(Point, Size)  // upper-left, dimensions
}
~~~

Now we might write a function to compute the area of a shape. This
function takes a reference to a shape, to avoid the need for
copying.

~~~
# struct Point {x: f64, y: f64}; // as before
# struct Size {w: f64, h: f64}; // as before
# enum Shape {
#     Circle(Point, f64),   // origin, radius
#     Rectangle(Point, Size)  // upper-left, dimensions
# }
# static tau: f64 = 6.28;
fn compute_area(shape: &Shape) -> f64 {
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
+-------+         |   {x: f64,    |
| size  | -+      |    y: f64},   |
+-------+  +----> |   {w: f64,    |
                  |    h: f64})   |
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
+-------+         |   {x: f64,    |
| size  | -+      |    y: f64},   |
+-------+  +----> |   f64)        |
                  |               |
                  +---------------+
~~~

As you can see, the `size` pointer would be pointing at a `f64`
instead of a struct. This is not good: dereferencing the second field
of a `f64` as if it were a struct with two fields would be a memory
safety violation.

So, in fact, for every `ref` binding, the compiler will impose the
same rules as the ones we saw for borrowing the interior of an owned
box: it must be able to guarantee that the `enum` will not be
overwritten for the duration of the borrow.  In fact, the compiler
would accept the example we gave earlier. The example is safe because
the shape pointer has type `&Shape`, which means "reference to
immutable memory containing a `shape`". If, however, the type of that
pointer were `&mut Shape`, then the ref binding would be ill-typed.
Just as with owned boxes, the compiler will permit `ref` bindings
into data owned by the stack frame even if the data are mutable,
but otherwise it requires that the data reside in immutable memory.

# Returning references

So far, all of the examples we have looked at, use references in a
“downward” direction. That is, a method or code block creates a
reference, then uses it within the same scope. It is also
possible to return references as the result of a function, but
as we'll see, doing so requires some explicit annotation.

We could write a subroutine like this:

~~~
struct Point {x: f64, y: f64}
fn get_x<'r>(p: &'r Point) -> &'r f64 { &p.x }
~~~

Here, the function `get_x()` returns a pointer into the structure it
was given. The type of the parameter (`&'r Point`) and return type
(`&'r f64`) both use a new syntactic form that we have not seen so
far.  Here the identifier `r` names the lifetime of the pointer
explicitly. So in effect, this function declares that it takes a
pointer with lifetime `r` and returns a pointer with that same
lifetime.

In general, it is only possible to return references if they
are derived from a parameter to the procedure. In that case, the
pointer result will always have the same lifetime as one of the
parameters; named lifetimes indicate which parameter that
is.

In the previous code samples, function parameter types did not include a
lifetime name. The compiler simply creates a fresh name for the lifetime
automatically: that is, the lifetime name is guaranteed to refer to a distinct
lifetime from the lifetimes of all other parameters.

Named lifetimes that appear in function signatures are conceptually
the same as the other lifetimes we have seen before, but they are a bit
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

~~~ {.ignore}
struct Point {x: f64, y: f64}
fn get_x_sh(p: &Point) -> &f64 {
    &p.x // Error reported here
}
~~~

Here, the function `get_x_sh()` takes a reference as input and
returns a reference. As before, the lifetime of the reference
that will be returned is a parameter (specified by the
caller). That means that `get_x_sh()` promises to return a reference
that is valid for as long as the caller would like: this is
subtly different from the first example, which promised to return a
pointer that was valid for as long as its pointer argument was valid.

Within `get_x_sh()`, we see the expression `&p.x` which takes the
address of a field of a Point. The presence of this expression
implies that the compiler must guarantee that , so long as the
resulting pointer is valid, the original Point won't be moved or changed.

But recall that `get_x_sh()` also promised to
return a pointer that was valid for as long as the caller wanted it to
be. Clearly, `get_x_sh()` is not in a position to make both of these
guarantees; in fact, it cannot guarantee that the pointer will remain
valid at all once it returns, as the parameter `p` may or may not be
live in the caller. Therefore, the compiler will report an error here.

In general, if you borrow a struct or box to create a
reference, it will only be valid within the function
and cannot be returned. This is why the typical way to return references
is to take references as input (the only other case in
which it can be legal to return a reference is if it
points at a static constant).

# Named lifetimes

Lifetimes can be named and referenced. For example, the special lifetime
`'static`, which does not go out of scope, can be used to create global
variables and communicate between tasks (see the manual for use cases).

## Parameter Lifetimes

Named lifetimes allow for grouping of parameters by lifetime.
For example, consider this function:

~~~
# struct Point {x: f64, y: f64}; // as before
# struct Size {w: f64, h: f64}; // as before
# enum Shape {
#     Circle(Point, f64),   // origin, radius
#     Rectangle(Point, Size)  // upper-left, dimensions
# }
# fn compute_area(shape: &Shape) -> f64 { 0.0 }
fn select<'r, T>(shape: &'r Shape, threshold: f64,
                 a: &'r T, b: &'r T) -> &'r T {
    if compute_area(shape) > threshold {a} else {b}
}
~~~

This function takes three references and assigns each the same
lifetime `r`.  In practice, this means that, in the caller, the
lifetime `r` will be the *intersection of the lifetime of the three
region parameters*. This may be overly conservative, as in this
example:

~~~
# struct Point {x: f64, y: f64}; // as before
# struct Size {w: f64, h: f64}; // as before
# enum Shape {
#     Circle(Point, f64),   // origin, radius
#     Rectangle(Point, Size)  // upper-left, dimensions
# }
# fn compute_area(shape: &Shape) -> f64 { 0.0 }
# fn select<'r, T>(shape: &Shape, threshold: f64,
#                  a: &'r T, b: &'r T) -> &'r T {
#     if compute_area(shape) > threshold {a} else {b}
# }
                                                     // -+ r
fn select_based_on_unit_circle<'r, T>(               //  |-+ B
    threshold: f64, a: &'r T, b: &'r T) -> &'r T {   //  | |
                                                     //  | |
    let shape = Circle(Point {x: 0., y: 0.}, 1.);    //  | |
    select(&shape, threshold, a, b)                  //  | |
}                                                    //  |-+
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
# struct Point {x: f64, y: f64}; // as before
# struct Size {w: f64, h: f64}; // as before
# enum Shape {
#     Circle(Point, f64),   // origin, radius
#     Rectangle(Point, Size)  // upper-left, dimensions
# }
# fn compute_area(shape: &Shape) -> f64 { 0.0 }
fn select<'r, 'tmp, T>(shape: &'tmp Shape, threshold: f64,
                       a: &'r T, b: &'r T) -> &'r T {
    if compute_area(shape) > threshold {a} else {b}
}
~~~

Here you can see that `shape`'s lifetime is now named `tmp`. The
parameters `a`, `b`, and the return value all have the lifetime `r`.
However, since the lifetime `tmp` is not returned, it would be more
concise to just omit the named lifetime for `shape` altogether:

~~~
# struct Point {x: f64, y: f64}; // as before
# struct Size {w: f64, h: f64}; // as before
# enum Shape {
#     Circle(Point, f64),   // origin, radius
#     Rectangle(Point, Size)  // upper-left, dimensions
# }
# fn compute_area(shape: &Shape) -> f64 { 0.0 }
fn select<'r, T>(shape: &Shape, threshold: f64,
                 a: &'r T, b: &'r T) -> &'r T {
    if compute_area(shape) > threshold {a} else {b}
}
~~~

This is equivalent to the previous definition.

## Labeled Control Structures

Named lifetime notation can also be used to control the flow of execution:

~~~
'h: for i in range(0,10) {
    'g: loop {
        if i % 2 == 0 { continue 'h; }
        if i == 9 { break 'h; }
        break 'g;
    }
}
~~~

> *Note:* Labelled breaks are not currently supported within `while` loops.

Named labels are hygienic and can be used safely within macros.
See the macros guide section on hygiene for more details.

# Conclusion

So there you have it: a (relatively) brief tour of the lifetime
system. For more details, we refer to the (yet to be written) reference
document on references, which will explain the full notation
and give more examples.
