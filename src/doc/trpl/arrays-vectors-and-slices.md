% Arrays, Vectors, and Slices

Like many programming languages, Rust has list types to represent a sequence of
things. The most basic is the **array**, a fixed-size list of elements of the
same type. By default, arrays are immutable.

```{rust}
let a = [1, 2, 3];     // a: [i32; 3]
let mut m = [1, 2, 3]; // mut m: [i32; 3]
```

There's a shorthand for initializing each element of an array to the same
value. In this example, each element of `a` will be initialized to `0`:

```{rust}
let a = [0; 20]; // a: [i32; 20]
```

Arrays have type `[T; N]`. We'll talk about this `T` notation later, when we
cover generics.

You can get the number of elements in an array `a` with `a.len()`, and use
`a.iter()` to iterate over them with a for loop. This code will print each
number in order:

```{rust}
let a = [1, 2, 3];

println!("a has {} elements", a.len());
for e in a.iter() {
    println!("{}", e);
}
```

You can access a particular element of an array with **subscript notation**:

```{rust}
let names = ["Graydon", "Brian", "Niko"]; // names: [&str; 3]

println!("The second name is: {}", names[1]);
```

Subscripts start at zero, like in most programming languages, so the first name
is `names[0]` and the second name is `names[1]`. The above example prints
`The second name is: Brian`. If you try to use a subscript that is not in the
array, you will get an error: array access is bounds-checked at run-time. Such
errant access is the source of many bugs in other systems programming
languages.

A **vector** is a dynamic or "growable" array, implemented as the standard
library type [`Vec<T>`](../std/vec/) (we'll talk about what the `<T>` means
later). Vectors are to arrays what `String` is to `&str`. You can create them
with the `vec!` macro:

```{rust}
let v = vec![1, 2, 3]; // v: Vec<i32>
```

(Notice that unlike the `println!` macro we've used in the past, we use square
brackets `[]` with `vec!`. Rust allows you to use either in either situation,
this is just convention.)

You can get the length of, iterate over, and subscript vectors just like
arrays. In addition, (mutable) vectors can grow automatically:

```{rust}
let mut nums = vec![1, 2, 3]; // mut nums: Vec<i32>

nums.push(4);

println!("The length of nums is now {}", nums.len());   // Prints 4
```

Vectors have many more useful methods.

A **slice** is a reference to (or "view" into) an array. They are useful for
allowing safe, efficient access to a portion of an array without copying. For
example, you might want to reference just one line of a file read into memory.
By nature, a slice is not created directly, but from an existing variable.
Slices have a length, can be mutable or not, and in many ways behave like
arrays:

```{rust}
let a = [0, 1, 2, 3, 4];
let middle = a.slice(1, 4);     // A slice of a: just the elements [1,2,3]

for e in middle.iter() {
    println!("{}", e);          // Prints 1, 2, 3
}
```

You can also take a slice of a vector, `String`, or `&str`, because they are
backed by arrays. Slices have type `&[T]`, which we'll talk about when we cover
generics.

We have now learned all of the most basic Rust concepts. We're ready to start
building our guessing game, we just need to know one last thing: how to get
input from the keyboard. You can't have a guessing game without the ability to
guess!
