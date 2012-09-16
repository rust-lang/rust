% Rust Borrowed Pointers Tutorial

# Introduction

Borrowed pointers are one of the more flexible and powerful tools
available in Rust. A borrowed pointer can be used to point anywhere:
into the shared and exchange heaps, into the stack, and even into the
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

As an example, consider a simple record type `point`:

~~~
type point = {x: float, y: float};
~~~

We can use this simple definition to allocate points in many ways. For
example, in this code, each of these three local variables contains a
point, but allocated in a different place:

~~~
let on_the_stack : point  =  {x: 3.0, y: 4.0};
let shared_box   : @point = @{x: 5.0, y: 1.0};
let unique_box   : ~point = ~{x: 7.0, y: 9.0};
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
fn compute_distance(p1: &point, p2: &point) -> float {
    let x_d = p1.x - p2.x;
    let y_d = p1.y - p2.y;
    sqrt(x_d * x_d + y_d * y_d)
}
~~~

Now we can call `compute_distance()` in various ways:

~~~
compute_distance(&on_the_stack, shared_box)
compute_distance(shared_box, unique_box)
~~~

Here the `&` operator is used to take the address of the variable
`on_the_stack`; this is because `on_the_stack` has the type `point`
(that is, a record value) and we have to take its address to get a
value. We also call this _borrowing_ the local variable
`on_the_stack`, because we are created an alias: that is, another
route to the same data.

In the case of the boxes `shared_box` and `unique_box`, however, no
explicit action is necessary. The compiler will automatically convert
a box like `@point` or `~point` to a borrowed pointer like
`&point`. This is another form of borrowing; in this case, the
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
let on_the_stack : point = {x: 3.0, y: 4.0};
~~~

This results in a by-value variable. As a consequence, we had to
explicitly take the address of `on_the_stack` to get a borrowed
pointer. Sometimes however it is more convenient to move the &
operator into the definition of `on_the_stack`:

~~~
let on_the_stack2 : &point = &{x: 3.0, y: 4.0};
~~~

Applying `&` to an rvalue (non-assignable location) is just a convenient
shorthand for creating a temporary and taking its address:

~~~
let tmp = {x: 3.0, y: 4.0};
let on_the_stack2 : &point = &tmp;
~~~

Taking the address of fields

As in C, the `&` operator is not limited to taking the address of
local variables. It can also be used to take the address of fields or
individual array elements. For example, consider this type definition
for `rectangle`:

~~~
type point = {x: float, y: float}; // as before
type size = {w: float, h: float}; // as before
type rectangle = {origin: point, size: size};
~~~

Now again I can define rectangles in a few different ways:

~~~
let rect_stack  = &{origin: {x: 1, y: 2}, size: {w: 3, h: 4}};
let rect_shared = @{origin: {x: 3, y: 4}, size: {w: 3, h: 4}};
let rect_unique = ~{origin: {x: 5, y: 6}, size: {w: 3, h: 4}};
~~~

In each case I can use the `&` operator to extact out individual
subcomponents. For example, I could write:

~~~
compute_distance(&rect_stack.origin, &rect_shared.origin);
~~~

which would borrow the field `origin` from the rectangle on the stack
from the shared box and then compute the distance between them.

# Borrowing shared boxes and rooting

We’ve seen a few examples so far where heap boxes (both shared and
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
fn example1() {
    let mut x = {f: 3};
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
fn example2() {
    let mut x = @{f: 3};
    let y = &x.f;      // -+ L
    ...                //  |
}                      // -+
~~~

In this example, the value `x` is in fact a heap box, and `y` is
therefore a pointer into that heap box. Again the lifetime of `y` will
be L, the remainder of the function body. But there is a crucial
difference: suppose `x` were reassigned during the lifetime L? If
we’re not careful, that could mean that the shared box would become
unrooted and therefore be subject to garbage collection

> ***Note:***In our current implementation, the garbage collector is
> implemented using reference counting and cycle detection.

For this reason, whenever the interior of a shared box stored in a
mutable location is borrowed, the compiler will insert a temporary
that ensures that the shared box remains live for the entire
lifetime. So, the above example would be compiled as:

~~~
fn example2() {
    let mut x = @{f: 3};
    let x1 = x;
    let y = &x1.f;     // -+ L
    ...                //  |
}                      // -+
~~~

Now if `x` is reassigned, the pointer `y` will still remain valid. This
process is called “rooting”.

# Borrowing unique boxes

The previous example demonstrated `rooting`, the process by which the
compiler ensures that shared boxes remain live for the duration of a
borrow. Unfortunately, rooting does not work if the data being
borrowed is a unique box, as it is not possible to have two references
to a unique box.

For unique boxes, therefore, the compiler will only allow a borrow `if
the compiler can guarantee that the unique box will not be reassigned
or moved for the lifetime of the pointer`. This does not necessarily
mean that the unique box is stored in immutable memory. For example,
the following function is legal:

~~~
fn example3() -> int {
    let mut x = ~{f: 3};
    if some_condition {
        let y = &x.f;      // -+ L
        ret *y;            //  |
    }                      // -+
    x = ~{f: 4};
    ...
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

~~~
fn example3() -> int {
    let mut x = ~{f: 3};
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
additional unique pointers and records, and the compiler will still be
able to detect possible mutations:

~~~
fn example3() -> int {
    let mut x = ~{mut f: ~{g: 3}};
    let y = &x.f.g;
    x = ~{mut f: ~{g: 4}}; // Error reported here.
    x.f = ~{g: 5};         // Error reported here.
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
fn example5a(x: @{mut f: ~{g: int}}, ...) -> int {
    let y = &x.f.g;   // Error reported here.
    ...
}
~~~

Here the heap looks something like:

~~~ {.notrust}
     Stack            Shared Heap       Exchange Heap

  x +------+        +-------------+       +------+
    | @... | ---->  | mut f: ~... | --+-> | g: 3 |
  y +------+        +-------------+   |   +------+
    | &int | -------------------------+
    +------+
~~~

In this case, the owning reference to the value being borrowed is in
fact `x.f`. Moreover, `x.f` is both mutable and aliasable. Aliasable
means that it is possible that there are other pointers to that same
shared box, so even if the compiler were to prevent `x.f` from being
mutated, the field might still be changed through some alias of
`x`. Therefore, to be safe, the compiler only accepts pure actions
during the lifetime of `y`. We’ll have a final example on purity but
inn unique fields, as in the following example:

Besides ensuring purity, the only way to borrow the interior of a
unique found in aliasable memory is to ensure that it is stored within
unique fields, as in the following example:

~~~
fn example5b(x: @{f: ~{g: int}}, ...) -> int {
    let y = &x.f.g;
    ...
}
~~~

Here, the field `f` is not declared as mutable. But that is enough for
the compiler to know that, even if aliases to `x` exist, the field `f`
cannot be changed and hence the unique box `g` will remain valid.

If you do have a unique box in a mutable field, and you wish to borrow
it, one option is to use the swap operator to bring that unique box
onto your stack:

~~~
fn example5c(x: @{mut f: ~int}, ...) -> int {
    let mut v = ~0;
    v <-> x.f;         // Swap v and x.f
    let y = &v;
    ...
    x.f <- v;          // Replace x.f
}
~~~

Of course, this has the side effect of modifying your shared box for
the duration of the borrow, so it works best when you know that you
won’t be accessing that same box again.

# Borrowing and enums

The previous example showed that borrowing unique boxes found in
aliasable, mutable memory is not permitted, so as to prevent pointers
into freed memory. There is one other case where the compiler must be
very careful to ensure that pointers remain valid: pointers into the
interior of an enum.

As an example, let’s look at the following `shape` type that can
represent both rectangles and circles:

~~~
type point = {x: float, y: float}; // as before
type size = {w: float, h: float}; // as before
enum shape {
    circle(point, float),   // origin, radius
    rectangle(point, size)  // upper-left, dimensions
}
~~~

Now I might write a function to compute the area of a shape. This
function takes a borrowed pointer to a shape to avoid the need of
copying them.

~~~
fn compute_area(shape: &shape) -> float {
    alt *shape {
        circle(_, radius) => 0.5 * tau * radius * radius,
        rectangle(_, ref size) => size.w * size.h
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
not a record. This is not good.

So, in fact, for every `ref` binding, the compiler will impose the
same rules as the ones we saw for borrowing the interior of a unique
box: it must be able to guarantee that the enum will not be
overwritten for the duration of the borrow.  In fact, the example I
gave earlier would be considered safe. This is because the shape
pointer has type `&shape`, which means “borrowed pointer to immutable
memory containing a shape”. If however the type of that pointer were
`&const shape` or `&mut shape`, then the ref binding would not be
permitted. Just as with unique boxes, the compiler will permit ref
bindings into data owned by the stack frame even if it is mutable, but
otherwise it requires that the data reside in immutable memory.

> ***Note:*** Right now, all pattern bindings are by-reference. We
> expect this to change so that copies are the default and references
> must be noted explicitly.

# Returning borrowed pointers

So far, all of the examples we’ve looked at use borrowed pointers in a
“downward” direction. That is, the borrowed pointer is created and
then used during the method or code block which created it. In some
cases, it is also possible to return borrowed pointers to the caller,
but as we’ll see this is more limited.

For example, we could write a subroutine like this:

~~~
type point = {x: float, y: float};
fn get_x(p: &point) -> &float { &p.x }
~~~

Here, the function `get_x()` returns a pointer into the structure it was
given. You’ll note that _both_ the parameter and the return value are
borrowed pointers; this is important. In general, it is only possible
to return borrowed pointers if they are derived from a borrowed
pointer which was given as input to the procedure.

In the example, `get_x()` took a borrowed pointer to a `point` as
input. In general, for all borrowed pointers that appear in the
signature of a function (such as the parameter and return types), the
compiler assigns the same symbolic lifetime L (we will see later that
there are ways to differentiate the lifetimes of different parameters
if that should be necessary). This means that, from the compiler’s
point of view, `get_x()` takes and returns two pointers with the same
lifetime. Now, unlike other lifetimes, this lifetime is a bit
abstract: it doesn’t refer to a specific expression within `get_x()`,
but rather to some expression within the caller. This is called a
_lifetime parameter_, because the lifetime L is effectively defined by
the caller to `get_x()`, just as the value for the parameter `p` is
defined by the caller.

In any case, whatever the lifetime L is, the pointer produced by
`&p.x` always has the same lifetime as `p` itself, as a pointer to a
field of a record is valid as long as the record is valid. Therefore,
the compiler is satisfied with the function `get_x()`.

To drill in this point, let’s look at a variation on the example, this
time one which does not compile:

~~~
type point = {x: float, y: float};
fn get_x_sh(p: @point) -> &float {
    &p.x // Error reported here
}
~~~

Here, the function `get_x_sh()` takes a shared box as input and
returns a borrowed pointer. As before, the lifetime of the borrowed
pointer that will be returned is a parameter (specified by the
caller). That means that effectively `get_x_sh()` is promising to
return a borrowed pointer that is valid for as long as the caller
would like: this is subtly different from the first example, which
promised to return a pointer that was valid for as long as the pointer
it was given.

Within `get_x_sh()`, we see the expression `&p.x` which takes the
address of a field of a shared box. This implies that the compiler
must guarantee that, so long as the resulting pointer is valid, the
shared box will not be reclaimed by the garbage collector. But recall
that get_x_sh() also promised to return a pointer that was valid for
as long as the caller wanted it to be. Clearly, `get_x_sh()` is not in
a position to make both of these guarantees; in fact, it cannot
guarantee that the pointer will remain valid at all once it returns,
as the parameter `p` may or may not be live in the caller. Therefore,
the compiler will report an error here.

In general, if you borrow a shared (or unique) box to create a
borrowed pointer, the pointer will only be valid within the function
and cannot be returned. Generally, the only way to return borrowed
pointers is to take borrowed pointers as input.

# Named lifetimes

So far we have always used the notation `&T` for a borrowed
pointer. However, sometimes if a function takes many parameters, it is
useful to be able to group those parameters by lifetime. For example,
consider this function:

~~~
fn select<T>(shape: &shape, threshold: float,
             a: &T, b: &T) -> &T {
    if compute_area(shape) > threshold {a} else {b}
}
~~~

This function takes three borrowed pointers. Because of the way that
the system works, each will be assigned the same lifetime: the default
lifetime parameter. In practice, this means that, in the caller, the
lifetime of the returned value will be the intersection of the
lifetime of the three region parameters. This may be overloy
conservative, as in this example:

~~~
                                              // -+ L
fn select_based_on_unit_circle<T>(            //  |-+ B
    threshold: float, a: &T, b: &T) -> &T {   //  | |
                                              //  | |
    let shape = circle({x: 0, y: 0}, 1);      //  | |
    select(&shape, threshold, a, b)           //  | |
}                                             //  |-+
                                              // -+
~~~

In this call to `select()`, the lifetime of the first parameter shape
is B, the function body. Both of the second two parameters `a` and `b`
share the same lifetime, L, which is the lifetime parameter of
`select_based_on_unit_circle()`. The caller will infer the
intersection of these three lifetimes as the lifetime of the returned
value, and hence the return value of `shape()` will be assigned a
return value of B. This will in turn lead to a compilation error,
because `select_based_on_unit_circle()` is supposed to return a value
with the lifetime L.

To address this, we could modify the definition of `select()` to
distinguish the lifetime of the first parameter from the lifetime of
the latter two. After all, the first parameter is not being
returned. To do so, we make use of the notation `&lt/T`, which is a
borrowed pointer with an explicit lifetime. This effectively creates a
second lifetime parameter for the function; named lifetime parameters
do not need to be declared, you just use them. Here is how the new
`select()` might look:

~~~
fn select<T>(shape: &tmp/shape, threshold: float,
             a: &T, b: &T) -> &T {
    if compute_area(shape) > threshold {a} else {b}
}
~~~

Here you can see the lifetime of shape is now being called `tmp`. The
parameters `a`, `b`, and the return value all remain with the default
lifetime parameter.

You could also write `select()` using all named lifetime parameters,
which might look like:

~~~
fn select<T>(shape: &tmp/shape, threshold: float,
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
fn example5a(x: @{mut f: ~{g: int}}, ...) -> int {
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
fn example5a(x: @{mut f: ~{g: int}}, ...) -> int {
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