% Vectors

A ‘vector’ is a dynamic or ‘growable’ array, implemented as the standard
library type [`Vec<T>`][vec]. The `T` means that we can have vectors
of any type (see the chapter on [generics][generic] for more).
Vectors always allocate their data on the heap.
You can create them with the `vec!` macro:

```rust
let v = vec![1, 2, 3, 4, 5]; // v: Vec<i32>
```

(Notice that unlike the `println!` macro we’ve used in the past, we use square
brackets `[]` with `vec!` macro. Rust allows you to use either in either situation,
this is just convention.)

There’s an alternate form of `vec!` for repeating an initial value:

```rust
let v = vec![0; 10]; // ten zeroes
```

## Accessing elements

To get the value at a particular index in the vector, we use `[]`s:

```rust
let v = vec![1, 2, 3, 4, 5];

println!("The third element of v is {}", v[2]);
```

The indices count from `0`, so the third element is `v[2]`.

It’s also important to note that you must index with the `usize` type:

```ignore
let v = vec![1, 2, 3, 4, 5];

let i: usize = 0;
let j: i32 = 0;

// works
v[i];

// doesn’t
v[j];
```

Indexing with a non-`usize` type gives an error that looks like this:

```text
error: the trait `core::ops::Index<i32>` is not implemented for the type
`collections::vec::Vec<_>` [E0277]
v[j];
^~~~
note: the type `collections::vec::Vec<_>` cannot be indexed by `i32`
error: aborting due to previous error
```

There’s a lot of punctuation in that message, but the core of it makes sense:
you cannot index with an `i32`.

## Out-of-bounds Access

If you try to access an index that doesn’t exist:

```ignore
let v = vec![1, 2, 3];
println!("Item 7 is {}", v[7]);
```

then the current thread will [panic] with a message like this:

```text
thread '<main>' panicked at 'index out of bounds: the len is 3 but the index is 7'
```

If you want to handle out-of-bounds errors without panicking, you can use
methods like [`get`][get] or [`get_mut`][get_mut] that return `None` when
given an invalid index:

```rust
let v = vec![1, 2, 3];
match v.get(7) {
    Some(x) => println!("Item 7 is {}", x),
    None => println!("Sorry, this vector is too short.")
}
```

## Iterating

Once you have a vector, you can iterate through its elements with `for`. There
are three versions:

```rust
let mut v = vec![1, 2, 3, 4, 5];

for i in &v {
    println!("A reference to {}", i);
}

for i in &mut v {
    println!("A mutable reference to {}", i);
}

for i in v {
    println!("Take ownership of the vector and its element {}", i);
}
```

Vectors have many more useful methods, which you can read about in [their
API documentation][vec].

[vec]: ../std/vec/index.html
[generic]: generics.html
[panic]: concurrency.html#panics
[get]: http://doc.rust-lang.org/std/vec/struct.Vec.html#method.get
[get_mut]: http://doc.rust-lang.org/std/vec/struct.Vec.html#method.get_mut
