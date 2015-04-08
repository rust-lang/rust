% Slices

A *slice* is a reference to (or "view" into) an array. They are useful for
allowing safe, efficient access to a portion of an array without copying. For
example, you might want to reference just one line of a file read into memory.
By nature, a slice is not created directly, but from an existing variable.
Slices have a length, can be mutable or not, and in many ways behave like
arrays:

```{rust}
let a = [0, 1, 2, 3, 4];
let middle = &a[1..4]; // A slice of a: just the elements 1, 2, and 3

for e in middle.iter() {
    println!("{}", e); // Prints 1, 2, 3
}
```

You can also take a slice of a vector, `String`, or `&str`, because they are
backed by arrays. Slices have type `&[T]`, which we'll talk about when we cover
generics.
