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

<!-- 

FIXME: I think I need to talk about USE here
    extern mod std;
    use std::prelude::*;

see Rust for Rubyists

-->


~~~~
fn main() {
    use std::io::print;

    print("hello?");
}
~~~~

To add variables and formatting to the printed output, use `print!` or `println!`.  The variables are inserted into the output at the `{}` placeholders.  FIXME second println to show decimal places formatting?

~~~~
fn main() {
    let n = 3.0; 
    let l = 'b';

    println!("hello? the number is {} and the letter is {}.", n, l);

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

# `struct`

So far we've look at simple variables and strings. To represent compound data, use structures, which are groupings of types, and are introduced by the `struct` keyword. For example, a structure that represents a point on a two dimensional grid is composed of two integers, `x` and `y`. You can access the types within the `struct` using dot notation. 

Create a point using a *struct literal* expression `let p = Point { x: 1, y: 2 };`, which assigns the point into the local variable `p`. The data for the point is stored directly on the stack, there is no heap allocation or pointer indirection, which improves performance.

~~~~
struct Point { x: int, y: int }
fn main() {
    let  p = Point { x: 1, y: 2};
    println!("{}", p.x); // prints 1
    
}
~~~~

In other words, if you look at the stack frame for the function `main()`, it looks like this:

~~~~
+------+
| ...  | --+
| 1    |   | p.x
| 2    |   | p.y
+------+ --+
~~~~

<!--
Those of you who are familiar with C and C++ will find this behavior familiar. In contrast, languages like Java or Ruby always store structures in the heap. This means that the stack frame would look something like this:

```
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

Storing aggregate data inline (ie, one the stack, with no pointer indirection) is critical for improving performance, because cache, malloc/free expensive, pointer chasing and cache invalidation require a complex runtime with garbage collector.

Rust does not introduce pointers unless you specifically specify them. If we define a type `Line` that consists of two `Point`s:

```
struct Line {
    p1: Point,
    p2: Point
}
```

The two `Point`s are laid out inline inside the `Line` struct. 

<!-- FIXME: not really sure how this should look -->

~~~~
+------+
| ...  | --+
| 1    |   | p.x
| 2    |   | p.y
+------+ --+
+------+
| ...  | --+
| 1    |   | p.x
| 2    |   | p.y
+------+ --+
~~~~

<!-- FIXME: reword

 In this way Rust can build up complex aggregate structures with simple storage requirements. (ugh todo fixme)

-->

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

-->

[guide-tup]: guide-syntax.html/tuples

# Stack vs. Heap

Rust [variables][guide-var] are allocated on the stack by default. Rust also lets you allocate variables on the heap, using the `~` symbol. To access the variable use the `*` symbol. 	Heap memory is freed when the variable goes out of scope. 

<!--  FIXME* and & @? operator? more info?  BOXES? -->

~~~~
fn main() {
    let mut x = 3; // allocated on the stack
    x = x + 1;
    
    let mut y = ~5; // allocated on the heap
    // y = y + 1; // Does not work
    *y = *y + 1;
    
}
~~~~



<!-- FIXME: rewrite this 

Many modern languages represent values as pointers to heap memory by default. In contrast, Rust, like C and C++, represents such types directly. Another way to say this is that aggregate data in Rust are unboxed. 

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
