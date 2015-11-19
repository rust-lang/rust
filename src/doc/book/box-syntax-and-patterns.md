%  Box Syntax and Patterns

Currently the only stable way to create a `Box` is via the `Box::new` method.
Also it is not possible in stable Rust to destructure a `Box` in a match
pattern. The unstable `box` keyword can be used to both create and destructure
a `Box`. An example usage would be:

```rust
#![feature(box_syntax, box_patterns)]

fn main() {
    let b = Some(box 5);
    match b {
        Some(box n) if n < 0 => {
            println!("Box contains negative number {}", n);
        },
        Some(box n) if n >= 0 => {
            println!("Box contains non-negative number {}", n);
        },
        None => {
            println!("No box");
        },
        _ => unreachable!()
    }
}
```

Note that these features are currently hidden behind the `box_syntax` (box
creation) and `box_patterns` (destructuring and pattern matching) gates
because the syntax may still change in the future.

# Returning Pointers

In many languages with pointers, you'd return a pointer from a function
so as to avoid copying a large data structure. For example:

```rust
struct BigStruct {
    one: i32,
    two: i32,
    // etc
    one_hundred: i32,
}

fn foo(x: Box<BigStruct>) -> Box<BigStruct> {
    Box::new(*x)
}

fn main() {
    let x = Box::new(BigStruct {
        one: 1,
        two: 2,
        one_hundred: 100,
    });

    let y = foo(x);
}
```

The idea is that by passing around a box, you're only copying a pointer, rather
than the hundred `i32`s that make up the `BigStruct`.

This is an antipattern in Rust. Instead, write this:

```rust
#![feature(box_syntax)]

struct BigStruct {
    one: i32,
    two: i32,
    // etc
    one_hundred: i32,
}

fn foo(x: Box<BigStruct>) -> BigStruct {
    *x
}

fn main() {
    let x = Box::new(BigStruct {
        one: 1,
        two: 2,
        one_hundred: 100,
    });

    let y: Box<BigStruct> = box foo(x);
}
```

This gives you flexibility without sacrificing performance.

You may think that this gives us terrible performance: return a value and then
immediately box it up ?! Isn't this pattern the worst of both worlds? Rust is
smarter than that. There is no copy in this code. `main` allocates enough room
for the `box`, passes a pointer to that memory into `foo` as `x`, and then
`foo` writes the value straight into the `Box<T>`.

This is important enough that it bears repeating: pointers are not for
optimizing returning values from your code. Allow the caller to choose how they
want to use your output.
