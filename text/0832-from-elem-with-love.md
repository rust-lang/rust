- Feature Name: direct to stable, because it modifies a stable macro
- Start Date: 2015-02-11
- RFC PR: https://github.com/rust-lang/rfcs/pull/832
- Rust Issue: https://github.com/rust-lang/rust/issues/22414

# Summary

Add back the functionality of `Vec::from_elem` by improving the `vec![x; n]` sugar to work with Clone `x` and runtime `n`.

# Motivation

High demand, mostly. There are currently a few ways to achieve the behaviour of `Vec::from_elem(elem, n)`:

```
// #1
let vec = Vec::new();
for i in range(0, n) {
    vec.push(elem.clone())
}
```

```
// #2
let vec = vec![elem; n]
```

```
// #3
let vec = Vec::new();
vec.resize(elem, n);
```

```
// #4
let vec: Vec<_> = (0..n).map(|_| elem.clone()).collect()
```

```
// #5
let vec: Vec<_> = iter::repeat(elem).take(n).collect();
```

None of these quite match the convenience, power, and performance of:

```
let vec = Vec::from_elem(elem, n)
```

* `#1` is verbose *and* slow, because each `push` requires a capacity check.
* `#2` only works for a Copy `elem` and const `n`.
* `#3` needs a temporary, but should be otherwise identical performance-wise.
* `#4` and `#5` are considered verbose and noisy. They also need to clone one more
time than other methods *strictly* need to.

However the issues for `#2` are *entirely* artifical. It's simply a side-effect of
forwarding the impl to the identical array syntax. We can just make the code in the
`vec!` macro better. This naturally extends the compile-timey `[x; n]` array sugar
to the more runtimey semantics of Vec, without introducing "another way to do it".

`vec![100; 10]` is also *slightly* less ambiguous than `from_elem(100, 10)`,
because the `[T; n]` syntax is part of the language that developers should be
familiar with, while `from_elem` is just a function with arbitrary argument order.

`vec![x; n]` is also known to be 47% more sick-rad than `from_elem`, which was
of course deprecated to due its lack of sick-radness.

# Detailed design

Upgrade the current `vec!` macro to have the following definition:

```rust
macro_rules! vec {
    ($x:expr; $y:expr) => (
        unsafe {
            use std::ptr;
            use std::clone::Clone;

            let elem = $x;
            let n: usize = $y;
            let mut v = Vec::with_capacity(n);
            let mut ptr = v.as_mut_ptr();
            for i in range(1, n) {
                ptr::write(ptr, Clone::clone(&elem));
                ptr = ptr.offset(1);
                v.set_len(i);
            }

            // No needless clones
            if n > 0 {
                ptr::write(ptr, elem);
                v.set_len(n);
            }

            v
        }
    );
    ($($x:expr),*) => (
        <[_] as std::slice::SliceExt>::into_vec(
            std::boxed::Box::new([$($x),*]))
    );
    ($($x:expr,)*) => (vec![$($x),*])
}
```

(note: only the `[x; n]` branch is changed)

Which allows all of the following to work:

```
fn main() {
    println!("{:?}", vec![1; 10]);
    println!("{:?}", vec![Box::new(1); 10]);
    let n = 10;
    println!("{:?}", vec![1; n]);
}
```

# Drawbacks

Less discoverable than from_elem. All the problems that macros have relative to static methods.

# Alternatives

Just un-delete from_elem as it was.

# Unresolved questions

No.
