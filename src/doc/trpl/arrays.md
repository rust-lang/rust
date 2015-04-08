% Arrays

Like many programming languages, Rust has list types to represent a sequence of
things. The most basic is the *array*, a fixed-size list of elements of the
same type. By default, arrays are immutable.

```{rust}
let a = [1, 2, 3]; // a: [i32; 3]
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

You can access a particular element of an array with *subscript notation*:

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
