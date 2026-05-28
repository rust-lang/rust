The `main` function was incorrectly declared.

Erroneous code example:

```compile_fail,E0580
fn main(x: i32) { // error: main function has wrong type
    println!("{}", x);
}
```

The `main` function prototype should never take arguments.
Example:

```
fn main() {
    // your code
}
```

If you want to get command-line arguments, use `std::env::args`. To exit with a
specified exit code, use `std::process::exit`.
