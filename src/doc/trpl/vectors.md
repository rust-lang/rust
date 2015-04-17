% Vectors

A *vector* is a dynamic or "growable" array, implemented as the standard
library type [`Vec<T>`](../std/vec/) (Where `<T>` is a [Generic](./generics.md)
statement). Vectors always allocate their data on the heap. Vectors are to
[slices][slices] what [`String`][string] is to `&str`. You can
create them with the `vec!` macro:

```{rust}
let v = vec![1, 2, 3]; // v: Vec<i32>
```

[slices]: primitive-types.html#slices
[string]: strings.html

(Notice that unlike the `println!` macro we've used in the past, we use square
brackets `[]` with `vec!`. Rust allows you to use either in either situation,
this is just convention.)

There's an alternate form of `vec!` for repeating an initial value:

```
let v = vec![0; 10]; // ten zeroes
```

You can get the length of, iterate over, and subscript vectors just like
arrays. In addition, (mutable) vectors can grow automatically:

```{rust}
let mut nums = vec![1, 2, 3]; // mut nums: Vec<i32>

nums.push(4);

println!("The length of nums is now {}", nums.len()); // Prints 4
```

Vectors have many more useful methods.
