% The (New) Rust Language Tutorial

An introductory tutorial to the Rust programming language. The tutorial focuses on the strengths and unique aspects of the Rust language, and is intentionally brief. 

You should already be already familiar with programming concepts and a programming language from the C-family. The [syntax guide][syntax] is a reference to the nuts and bolts of the language, essentials such as [variables][guide-var], [functions][guide-fn], etc.

[syntax]: guide-syntax.html
[guide-var]: guide-syntax.html#variables
[guide-fn]: guide-syntax.html#functions

# Output and Input

Output to the console using `print`. Use the `println` convenience function instead of `print` to add a newline to the end of the output. Note the `use std::io::print` syntax to include the `print` function from the standard `io` library.

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

Structures are groupings of types, and are introduced by the `struct` keyword. For example, a structure that represents a point on a two dimensional grid is composed of two integers, `x` and `y`. Access the types within the `struct` using the name of the structure instance, a dot `.` and the name of the type inside the structure. In the following example, `p` is the instance of the `Point` structure and `p.x` and `p.y` are the coordinates, with values 1 and 2 respectively. 

~~~~
struct Point { x: int, y: int }
fn main() {
    let  p = Point { x: 1, y: 2};
    println!("{}", p.x); // prints 1
    
}
~~~~

FIXME: Should talk about Pointers first. 

Structures in Rust contain the values of the instances, not pointers to the values like most other languages in the C-family. If you copy a structure, you copy the entire structure, not just a pointer to the original structure.

~~~~
struct Point { x: int, y: int }

fn main() {
    let mut p = Point { x: 1, y: 2};
    let q = p; 
    p.x += 1; 
    println!("{}", q.x); // still prints 1
}
~~~~



<!-- 
FIXME:

* difference of stack vs heap
* values of aggregate type (Point is not a pointer to a Point)
* copying (not compared to moving
* to modify p.x, you make p mutable
  - don't explain inherited mutability in depth

-->

Tuples are structures of types without names, such a point composed of two unnamed integers `struct Point(int,int)`. See [tuples][guide-tup] for more information.

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

<!-- FIXME: rewrite this -->

Many modern languages represent values as pointers to heap memory by default. In contrast, Rust, like C and C++, represents such types directly. Another way to say this is that aggregate data in Rust are unboxed. * more examples of types inline with other types
  - e.g. Line { p1, p2 }
  - show efficiency of inline struct layout

~~~~
struct Point { x: int, y: int }

fn main() {
    let mut p = Point { x: 1, y: 2};
    let q = p; // copies p deeply
    p.x += 1; // legal because p is mutable
    println!("{}", q.x); // still prints 1
}
~~~~

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
