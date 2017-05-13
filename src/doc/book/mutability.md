% Mutability

Mutability, the ability to change something, works a bit differently in Rust
than in other languages. The first aspect of mutability is its non-default
status:

```rust,ignore
let x = 5;
x = 6; // Error!
```

We can introduce mutability with the `mut` keyword:

```rust
let mut x = 5;

x = 6; // No problem!
```

This is a mutable [variable binding][vb]. When a binding is mutable, it means
you’re allowed to change what the binding points to. So in the above example,
it’s not so much that the value at `x` is changing, but that the binding
changed from one `i32` to another.

[vb]: variable-bindings.html

You can also create a [reference][ref] to it, using `&x`, but if you want to use the reference to change it, you will need a mutable reference:

```rust
let mut x = 5;
let y = &mut x;
```

[ref]: references-and-borrowing.html

`y` is an immutable binding to a mutable reference, which means that you can’t bind 'y' to something else (`y = &mut z`), but `y` can be used to bind `x` to something else (`*y = 5`). A subtle distinction.

Of course, if you need both:

```rust
let mut x = 5;
let mut y = &mut x;
```

Now `y` can be bound to another value, and the value it’s referencing can be
changed.

It’s important to note that `mut` is part of a [pattern][pattern], so you
can do things like this:

```rust
let (mut x, y) = (5, 6);

fn foo(mut x: i32) {
# }
```

Note that here, the `x` is mutable, but not the `y`.

[pattern]: patterns.html

# Interior vs. Exterior Mutability

However, when we say something is ‘immutable’ in Rust, that doesn’t mean that
it’s not able to be changed: we are referring to its ‘exterior mutability’ that
in this case is immutable. Consider, for example, [`Arc<T>`][arc]:

```rust
use std::sync::Arc;

let x = Arc::new(5);
let y = x.clone();
```

[arc]: ../std/sync/struct.Arc.html

When we call `clone()`, the `Arc<T>` needs to update the reference count. Yet
we’ve not used any `mut`s here, `x` is an immutable binding, and we didn’t take
`&mut 5` or anything. So what gives?

To understand this, we have to go back to the core of Rust’s guiding
philosophy, memory safety, and the mechanism by which Rust guarantees it, the
[ownership][ownership] system, and more specifically, [borrowing][borrowing]:

> You may have one or the other of these two kinds of borrows, but not both at
> the same time:
>
> * one or more references (`&T`) to a resource,
> * exactly one mutable reference (`&mut T`).

[ownership]: ownership.html
[borrowing]: references-and-borrowing.html#borrowing

So, that’s the real definition of ‘immutability’: is this safe to have two
pointers to? In `Arc<T>`’s case, yes: the mutation is entirely contained inside
the structure itself. It’s not user facing. For this reason, it hands out `&T`
with `clone()`. If it handed out `&mut T`s, though, that would be a problem.

Other types, like the ones in the [`std::cell`][stdcell] module, have the
opposite: interior mutability. For example:

```rust
use std::cell::RefCell;

let x = RefCell::new(42);

let y = x.borrow_mut();
```

[stdcell]: ../std/cell/index.html

RefCell hands out `&mut` references to what’s inside of it with the
`borrow_mut()` method. Isn’t that dangerous? What if we do:

```rust,ignore
use std::cell::RefCell;

let x = RefCell::new(42);

let y = x.borrow_mut();
let z = x.borrow_mut();
# (y, z);
```

This will in fact panic, at runtime. This is what `RefCell` does: it enforces
Rust’s borrowing rules at runtime, and `panic!`s if they’re violated. This
allows us to get around another aspect of Rust’s mutability rules. Let’s talk
about it first.

## Field-level mutability

Mutability is a property of either a borrow (`&mut`) or a binding (`let mut`).
This means that, for example, you cannot have a [`struct`][struct] with
some fields mutable and some immutable:

```rust,ignore
struct Point {
    x: i32,
    mut y: i32, // Nope.
}
```

The mutability of a struct is in its binding:

```rust,ignore
struct Point {
    x: i32,
    y: i32,
}

let mut a = Point { x: 5, y: 6 };

a.x = 10;

let b = Point { x: 5, y: 6};

b.x = 10; // Error: cannot assign to immutable field `b.x`.
```

[struct]: structs.html

However, by using [`Cell<T>`][cell], you can emulate field-level mutability:

```rust
use std::cell::Cell;

struct Point {
    x: i32,
    y: Cell<i32>,
}

let point = Point { x: 5, y: Cell::new(6) };

point.y.set(7);

println!("y: {:?}", point.y);
```

[cell]: ../std/cell/struct.Cell.html

This will print `y: Cell { value: 7 }`. We’ve successfully updated `y`.
