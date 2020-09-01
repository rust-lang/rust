A variable was borrowed as mutable more than once.

Erroneous code example:

```compile_fail,E0499
let mut i = 0;
let mut x = &mut i;
let mut a = &mut i;
x;
// error: cannot borrow `i` as mutable more than once at a time
```

Please note that in Rust, you can either have many immutable references, or one
mutable reference. For more details you may want to read the
[References & Borrowing][references-and-borrowing] section of the Book.

[references-and-borrowing]: https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html

Example:

```
let mut i = 0;
let mut x = &mut i; // ok!

// or:
let mut i = 0;
let a = &i; // ok!
let b = &i; // still ok!
let c = &i; // still ok!
b;
a;
```
