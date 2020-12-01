An attempt to use index on a type which doesn't implement the `std::ops::Index`
trait was performed.

Erroneous code example:

```compile_fail,E0608
0u8[2]; // error: cannot index into a value of type `u8`
```

To be able to index into a type it needs to implement the `std::ops::Index`
trait. Example:

```
let v: Vec<u8> = vec![0, 1, 2, 3];

// The `Vec` type implements the `Index` trait so you can do:
println!("{}", v[2]);
```
