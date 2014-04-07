% The (New) Rust Language Tutorial

An introductory tutorial to the Rust programming language. The tutorial focuses on Rust's strengths and unique design, and is intentionally brief.

You should already be already familiar with programming concepts, in particular pointers, and a programming language from the C-family. The [syntax guide][syntax] is a reference to the nuts and bolts of the language, essentials such as [variables][guide-var], [functions][guide-fn], [structs][guide-struct] etc.

<!-- FIXME link link to pointers? -->

[syntax]: guide-syntax.html
[guide-var]: guide-syntax.html#variables
[guide-fn]: guide-syntax.html#functions
[guide-struct]: guide-syntax.html#struct


# Output and Input

Output to the console using `print`. Use the `println` convenience function instead of `print` to add a newline to the end of the output. Note the `use std::io::print` syntax to include the `print` function from the standard `io` library. You can find more information on [use::][guide-use] in the [syntax guide][syntax].

[guide-use]: guide-syntax.html#use

~~~~
fn main() {
    use std::io::print;

    print("hello?");
}
~~~~

To add variables and formatting to the printed output, use `print!` or `println!`.  Variables are introduced with the `let` keyword, and unlike in most other languages are not changeable (or in Rust terms, mutable) by default. To make a  variable mutable, use `let mut`. More information on [variables][guide-var].

The variables are inserted into the output at the `{}` placeholders.

<!-- FIXME second println to show decimal places formatting?
    println!("hello? the number is {:f} and the letter is {}.", n, l);
-->

~~~~
fn main() {
    let n = 3.5;
    let l = 'b';

    println!("hello? the number is {} and the letter is {}.", n as int, l);

}
~~~~

For more information on formatting strings, read [`std::fmt`][fmt].

[fmt]: http://static.rust-lang.org/doc/0.9/std/fmt/index.html

Getting user input is slightly more involved, the following code takes each line of input, and prints it back after the string "You wrote:". Note the bracketed `use` on line one to include two different functions on the same line.

~~~~
use std::io::{stdin, BufferedReader};
fn main() {
  let mut stdin = BufferedReader::new(stdin());
  for line in stdin.lines() {
      println!("You wrote: {}", line);
  }
}
~~~~

# Stack and Heap, `struct`

So far we've looked at simple variables and strings. To represent compound data use structures, which are groupings of data types, and are introduced by the `struct` keyword. For example, a structure that represents a point on a two dimensional grid is composed of two integers, `x` and `y`. You can access the types within the `struct` using dot notation.

Create a point using a *struct literal* expression `let p = Point { x: 1, y: 2 };`, which assigns the point into the local variable `p`. The data for the point is stored directly on the stack, there is no heap allocation or pointer indirection, which improves performance.

<!-- FIXME: struct literal? -->

~~~~
struct Point { x: int, y: int }
fn main() {
    let  p = Point { x: 1, y: 2};
    println!("{}", p.x); // prints 1

}
~~~~

If you look at the stack frame for the function `main()`, it looks like this:

~~~~{.notrust}
+------+
| ...  | --+
| 1    |   | p.x
| 2    |   | p.y
+------+ --+
~~~~

<!--
Those of you who are familiar with C and C++ will find this behavior familiar. In contrast, languages like Java or Ruby always store structures in the heap. This means that the stack frame would look something like this:

```{.notrust}
(TODO: Make me not look like vomit)
-----------------     -----------------------------
| struct p            | 1
|   x: ---------------|
|   y: --------------------------
                                 -----------------------------
                                 | 2
                                 -----------------------------
```

-->

Storing aggregate data inline is critical for improving performance, because cache, malloc/free, pointer chasing and cache invalidation require a complex runtime with garbage collector.  If we define a type `Line` that consists of two `Point`s:

~~~~{.notrust}
struct Line {
    p1: Point,
    p2: Point
}
~~~~

The two `Point`s are similarly laid out inline inside the `Line` struct, which avoids the memory overhead of multiple headers and pointers to objects in different locations.

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

## Stack vs. Heap

<!-- FIXME: this is not correct right? BOX vs stack vs heap? -->

Rust does not introduce pointers to the heap unless you specify that you require them, using the `~` symbol. To access the contents of the pointer, use the `*` symbol to dereference it. Heap memory is freed when the variable goes out of scope.

~~~~
fn main() {
    let mut x = 3; // allocated on the stack
    x = x + 1;

    let mut y = ~5; // allocated on the heap
    // y = y + 1; // Does not work
    *y = *y + 1;

}
~~~~


<!--
FIXME:

~~~~
struct Point { x: int, y: int }

fn main() {
    let mut p = Point { x: 1, y: 2};
    let q = p;
    p.x += 1;
    println!("{}", q.x); // still prints 1
}
~~~~


* copying (not compared to moving
* to modify p.x, you make p mutable
  - don't explain inherited mutability in depth

-->
<!--

Tuples are structures of types without names, such a point composed of two unnamed integers `struct Point(int,int)`. See [tuples][guide-tup] for more information.


[guide-tup]: guide-syntax.html/tuples

-->



<!-- FIXME: rewrite this


~~~~
struct Point { x: int, y: int }

fn main() {
    let mut p = Point { x: 1, y: 2};
    let q = p; // copies p deeply
    p.x += 1; // legal because p is mutable
    println!("{}", q.x); // still prints 1
}
~~~~

-->

#  Move Vs Copy

<!-- FIXME: talk about this in a SIMPLE manner?

## `enum`

Enumerations are

### Enums and matching (???)

Introduce enums and matching here? Or after ownership? After 3.3?

-->

# Ownership

~~~~
struct Point {x: int, y: int}
fn main() {
    let mut p = ~Point { x: 1, y: 2 };
    let q = p;

    // p.x += 1; // error, huh?

}
~~~~~

* freeing

# Borrowing

Ownership is not the right tool when you just want temporary access to data:

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

## Immutable borrow

~~~~
struct Point {x: int, y: int}
fn get_point(p: &Point) -> int {
    p.x + p.y
}
~~~~

* only talking about using borrowed pointers in parameters
* make it explicit we're not talking about returning borrowed pointers
* don't introduce lifetime parameters yet
* don't put pointers into structs
