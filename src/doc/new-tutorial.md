% The (New) Rust Language Tutorial

# Stack vs. heap, values, copying

## Variables



~~~~
struct Point { x: int, y: int }
fn main() {
    let mut p = Point { x: 1, y: 2};
    let q = p; // copies p deeply
    p.x += 1; // legal because p is mutable
    println!("{}", q.x); // still prints 1
}
~~~~

* difference of stack vs heap
* values of aggregate type (Point is not a pointer to a Point)
* more examples of types inline with other types
  - e.g. Line { p1, p2 }
  - show efficiency of inline struct layout
* how to write a function
* do basic I/O `println!`
* copying (not compared to moving)
* to modify p.x, you make p mutable
  - don't explain inherited mutability in depth

## Using Functions

Functions in Rust are introduced with the `fn` keyword, optional parameters are specified within brackets as comma separated `name: type` pairs, and `->` indicates the return type. You can ommit the return type for functions that do not return a value. Functions return the top level expression (note the return expression is not terminated with a semi colon).

~~~~
fn main() {
    let i = 7;

    fn increment(i:int) -> (int) {
       i + 1 
    }	

    let k = increment(i); // k=8
}
~~~~

## Printing

This small program prints "hello?" to the console. Use `println` instead of `print` to add a newline to the end of the output.

~~~~
fn main() {
    use std::io::print;

    print("hello?");
}
~~~~

To add variables and formatting to the printed output, use `print!` or `println!`.  The variables are inserted into the output at the `{}` placeholders. 

~~~~
fn main() {
    let n = 3; 
    let l = 'b';

    println!("hello? the number is {} and the letter is {}.", n, l);
}
~~~~

For more information on formatting strings look at [`std::fmt`][fmt]. We explain what the `::` syntax means at FIXME.

[fmt]: http://static.rust-lang.org/doc/0.9/std/fmt/index.html

### Enums and matching (???)

Introduce enums and matching here? Or after ownership? After 3.3?

## Ownership (front and center)

~~~~ {.notrust}
struct Point {x: int, y: int}
fn main() {
    let mut p = ~Point { x: 1, y: 2 };
    let q = p;
    p.x += 1; // error, huh?
}
~~~~~

* heap allocation (~)
* moving
* freeing

## Borrowing

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
