% The Rust Tutorial 

Rust is a compiled programming language with a focus on type safety, memory safety, concurrency and performance. It has a sophisticated memory model that encourages efficient data structures and safe concurrency patterns, forbidding invalid memory accesses that would otherwise cause segmentation faults. 

This brief introduction to Rust focuses on Rust's design choices, strengths and 'philosophy'. You should already be already familiar with programming concepts, in particular pointers, and a programming language from the C-family. The [syntax guide][syntax] is a reference to the nuts and bolts of the language, essentials such as [variables][guide-var], [functions][guide-fn], [structs][guide-struct] etc.

[syntax]: guide-syntax.html
[guide-var]: guide-syntax.html#variables
[guide-fn]: guide-syntax.html#functions
[guide-struct]: guide-syntax.html#struct

# Running the Example Code

Before we get started, lets show you how to compile the code examples shown in the tutorial.

## Installing Rust

To install Rust on Windows:

1. Download and run the latest stable installer (.exe) from the [Rust website][install].

To install Rust on OS X:

1. Download and run the latest stable installer (.pkg) from the [Rust website][install].

To install Rust on Linux:

1. Download and extract the latest stable binaries (.tar.gz) from the [Rust website][install].
2. Run the `./install.sh` inside the directory.  

There are more complete [installation intructions][install-wiki] on the wiki.

[install]: http://www.rust-lang.org/install.html
[install-wiki]: https://github.com/mozilla/rust/wiki/Doc-packages,-editors,-and-other-tools

## Compiling and Running Rust Examples

1. Copy and paste the following source into an empty text file and save it as `example.rs`. By convention Rust source files have the `.rs` extension.
	
    ~~~~
    fn main() {
        println!("Hello world!");
    }
    ~~~~

2. Run `rustc example.rs`
3. If there are no compiler errors, run `"./example"` and see "Hello world!" printed to screen. 

# Stack, Heap and Inline Structures

To show how Rust semantics take maximum advantage of machine architecture to optimize runtime, we use a structure that represents a point in two dimensions: 

~~~~
struct Point {
    x: int,
    y: int
}
~~~~

Create a point using a *struct literal* expression which assigns the point into the local variable `p`, in this case specifying `x=1` and `y=2`.

~~~~
struct Point { x: int, y: int }

fn main() {
    let  p = Point { x: 1, y: 2};
}
~~~~

<!-- FIXME: more detail, help from @nikomatsakis? 

The data for the point is stored directly on the stack, there is no heap allocation or pointer indirection, which improves performance. 
-->

The stack frame for the function `main()` looks like this:

~~~~{.notrust}
+------+
| ...  | --+
| 1    |   | p.x
| 2    |   | p.y
+------+ --+
~~~~

Those of you who are familiar with C and C++ will find this behavior familiar. In contrast, languages like Java or Ruby always store structures in the heap, so the stack frame looks like this:

<!-- (TODO: fix this image)

https://github.com/mozilla/rust/pull/14017#discussion_r12404628


-->

~~~~{.notrust}
-----------------     -----------------------------
| struct p            | 1
|   x: ---------------|
|   y: --------------------------
                                 -----------------------------
                                 | 2
                                 -----------------------------
~~~~

<!-- FIXME: more detail, help from @nikomatsakis? 

Storing aggregate data inline is critical for improving performance, because malloc/free, pointer chasing and cache invalidation require a complex runtime with garbage collector.  
-->

Rust does not introduce pointers unless you specifically request them. If we define a type `Line` that consists of two `Point`s:

~~~~
struct Point {
    x: int,
    y: int
}
struct Line {
    p1: Point,
    p2: Point
}
~~~~

The two `Point`s are laid out inline inside the `Line` struct, which avoids the memory overhead of multiple headers and pointers to objects in different locations.

<!-- FIXME: not really sure how this should look -->

~~~~{.notrust}
+------------------+
|+------+          |
|| ...  | --+      |
|| 1    |   | p1.x |
|| 2    |   | p1.y |
|+------+ --+      |
|+------+          |
|| ...  | --+      |
|| 1    |   | p2.x |
|| 2    |   | p2.y |
|+------+ --+      |
+------------------+
~~~~

<!-- FIXME: here we continue talking about copying values as they are passed up and down the stack before introducing Rust's more common pattern - *moving* in the next section-->

Function parameters are laid out inline in the same way. A function that takes two points as arguments reserves space on the stack for the two point arguments:

~~~~{.notrust}
fn draw_line(from: Point, to: Point) { // Draw the line }
~~~~


## Heap Allocation

The stack is very efficient but it is also limited in size, so can't be used to store arbitrary amounts of data. To do that you need to allocate memory on the heap.

<!-- FIXME: not really happy with this, we're already talking about allocating memory, so the ~ is forced -->

In Rust, the box operator (`~`) allocates memory, so `~expr` allocates space on the heap, evaluates `expr` and stores the result on the heap. For example, `p=~Point { x: 1, y: 2 }`. `p` is not of type `Point` but of type `~Point`,  the `~` indicates that it is a pointer to a heap allocation.

One very common use for heap allocation is to store [arrays][arrays]. For example, a function to draw a polygon of an arbitrary list of points:

[arrays]:  FIXME

~~~~
struct Point {
    x: int,
    y: int
}

fn draw_line(from: Point, to: Point) { // Draw the line }

fn draw_polygon(points: ~[Point]) {
    let mut i = 0;
    while i < points.len() - 1 {
        draw_line(points[i], points[i+1]);
        i += 1;
    }
    draw_line(points[i], points[0]);
}
~~~~

The type `~[...]` in Rust indicates a heap-allocated array containing a variable number of points. Calling `draw_polygon`:

<!-- FIXME Reword? You can create a `~[...]` array using a `~[...]` expression.  -->

~~~~
# struct Point { x: int, y: int }
# fn draw_line(from: Point, to: Point) { // Draw the line }
fn draw_polygon(points: ~[Point]) {
    let mut i = 0;
    while i < points.len() - 1 {
        draw_line(points[i], points[i+1]);
        i += 1;
    }
    draw_line(points[i], points[0]);
}

fn main() { 
    let p1 = Point { x: 0, y: 0 }; 
    let p2 = Point { x: 0, y: 100 }; 
    let p3 = Point { x: 100, y: 100 }; 
    let points = ~[p1, p2, p3]; 
    draw_polygon(points); 
} 
~~~~

<!--

In Rust, whenever a heap pointer goes out of scope, the memory it points at is automatically freed.

 For example, when the `draw_polygon()` function returns, it automatically frees its argument, `points`.

 -->

# Ownership and Moving 

<!-- obective: show why ownership is important
               how moving works -->

<!--Now talk about draw_polygon and moving.-->

In other languages, manually freeing memory carries the risk that if you use a pointer after it has been freed, you are using memory that no longer belongs to you, which might have been reused elsewhere. The results are unpredictable, leading to errors, crashes, and security vulnerabilities.

Rust is designed to guarantee that once a pointer is freed, it can never be used again. Every heap pointer has an *owner*, and only the owner of a pointer may use it. When a pointer is first allocated, the owner is the function that allocated it. But when the pointer is passed as an argument, ownership is transferred to the callee. This means that the caller can no longer access the pointer or the memory it points at.

Revisiting the `draw_polygon` example:

~~~~
struct Point {
    x: int,
    y: int
}

fn draw_polygon(points: ~[Point]) { // Draw the line }

fn main() {
    // `points` is initially owned by `main()`
    let points = ~[...];
    
    // `points` is given to `draw_polygon()`
    draw_polygon(points);
    
    // Error: `points` has been given away, cannot access
    draw_line(points[0], points[1]);
}
~~~~

The last line, which attempts to use `points` after it has been *given* to `draw_polygon()` is flagged as an error by the Rust compiler. Because `draw_polygon()` has already freed the array's memory when it returned, this access would be reading from freed memory, which is not allowed.
 
Ownership can also be transferred by returning a value or by storing it an array or the field of a struct. For example, we can modify the `draw_polygon()` function to return the array of points at the end, so that the caller can go on using it 

<!-- (Note: you probably wouldn't actually write the code this way; in the next section, we'll introduce "borrowing", which allows you to give temporary access to an array without transferring ownership): -->

~~~~
# struct Point {
#     x: int,
#     y: int
# }

fn draw_polygon(points: ~[Point]) -> ~[Point] {
    return points;
}

fn main() {
    // `points` is initially owned by `main()`
    let points = ~[...];
    
    // `points` is given to `draw_polygon()` then taken back.
    let points = draw_polygon(points);
    
    // This is now ok
    draw_line(points[0], points[1]);
}
~~~~

<!-- now generalize ownership to other types;
     when are things movable vs. copyable -->

In Rust, owned pointers are called owned boxes, as they represent pointers to 'boxes' of memory. Ownership is not just about owned boxes. By default, whenever you declare a new type, values of that type are *moved* from place to place when you use them. By *move*, we mean that ownership of the value is transferred to its new location.

Ownership is a powerful tool for working with values that require some sort of cleanup or which have some sort of destructor. Examples are owned boxes, which need to be freed, or file handles, which need to be closed. Ownership tells us who will perform this cleanup.

For simple data types, however, this sort of tracking is inconvenient. The `Point` type that we introduced earlier is simply a pair of integers and requires no cleanup, so we might prefer that it be implicitly copied each time we use it rather than transferring ownership. To indicate that, we annotate the `Point` type with an attribute, `#[deriving(Copy)]`:
 

~~~~
#[deriving(Copy)]
struct Point {
    x: int,
    y: int,
}
~~~~

This annotation is an example of a Rust *attribute*. Rust attributes generally have the form `#[...]` and can be attached to any declaration. There are detailed guides on [attributes][attributes] in general and the [deriving][deriving] attribute in particular.

[attributes]: FIXME
[deriving]: FIXME

Now that we have indicated that `Point` values should be copied and not moved, we can continue using them even after they have been passed as parameters or stored into data structures:

~~~~
# struct Point {
#    x: int,
#    y: int
# }

fn main() {
    let point1 = Point { x:1, y:2 };
    let point2 = Point { x:5, y:6 };

    // Passing copies of some points to a function
    draw_line(point1, point2);
    
    // Inserting copies of the same points into a boxed array.
    // If `Point` were not declared as `#[deriving(Copy)]`,
    // the compiler would flag an error
    let points = ~[point1, point2];
    draw_polygon(points);
}
~~~~

# Borrowing

Ownership is not the right tool when you just want temporary access to data:

<!--

~~~~
struct Point {x: int, y: int}
fn main() {
    let p = ~Point { x: 1, y: 2};
    let r = inc_allocated_point(p);
}
fn inc_allocated_point(mut p: ~Point) -> ~Point {
    p.x += 1;
    p // final expression is returned
}
~~~~~

* final expression is returned (syntax note)
* writing a  function with arguments
* mutability on function arguments

~~~~
struct Point {x: int, y: int}
fn main() {
    let mut p = ~Point {x: 1, y: 2};
    inc_allocated_point(&mut p);
}
fn inc_allocated_point(p: &mut ~Point) {
    p.x += 1;
}
~~~~
-->

## Immutable borrow

<!--
~~~~
struct Point {x: int, y: int}
fn get_point(p: &Point) -> int {
    p.x + p.y
}
~~~~

-->

* only talking about using borrowed pointers in parameters
* make it explicit we're not talking about returning borrowed pointers
* don't introduce lifetime parameters yet
* don't put pointers into structs

# Mutability

# Lifetimes 

# Vectors vs. Slices 

# Structs, enums, and pattern matching
