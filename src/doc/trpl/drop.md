% Drop

Now that we’ve discussed traits, let’s talk about a particular trait provided
by the Rust standard library, [`Drop`][drop]. The `Drop` trait provides a way
to run some code when a value goes out of scope. For example:

[drop]: ../std/ops/trait.Drop.html

```rust
struct HasDrop;

impl Drop for HasDrop {
    fn drop(&mut self) {
        println!("Dropping!");
    }
}

fn main() {
    let x = HasDrop;

    // do stuff

} // x goes out of scope here
```

When `x` goes out of scope at the end of `main()`, the code for `Drop` will
run. `Drop` has one method, which is also called `drop()`. It takes a mutable
reference to `self`.

That’s it! The mechanics of `Drop` are very simple, but there are some
subtleties. For example, values are dropped in the opposite order they are
declared. Here’s another example:

```rust
struct Firework {
    strength: i32,
}

impl Drop for Firework {
    fn drop(&mut self) {
        println!("BOOM times {}!!!", self.strength);
    }
}

fn main() {
    let firecracker = Firework { strength: 1 };
    let tnt = Firework { strength: 100 };
}
```

This will output:

```text
BOOM times 100!!!
BOOM times 1!!!
```

The TNT goes off before the firecracker does, because it was declared
afterwards. Last in, first out. There’s one exception to this, however:
a binding to `_` will be dropped immediately:

```rust
struct Noisy(u8);
 
impl Drop for Noisy {
    fn drop(&mut self) { println!("dropping {}", self.0); }
}
 
fn main() {
    let _ = Noisy(1);
    let x = Noisy(2);
    let y = Noisy(3);
} 
```

Will print:

```text
dropping 1
dropping 3
dropping 2
```

rather than

```text
dropping 3
dropping 2
dropping 1
```

So what is `Drop` good for? Generally, `Drop` is used to clean up any resources
associated with a `struct`. For example, the [`Arc<T>` type][arc] is a
reference-counted type. When `Drop` is called, it will decrement the reference
count, and if the total number of references is zero, will clean up the
underlying value.

[arc]: ../std/sync/struct.Arc.html
