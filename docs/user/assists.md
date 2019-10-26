# Assists

## `add_derive`

Adds a new `#[derive()]` clause to a struct or enum.

```rust
// BEFORE
struct Point {
    x: u32,
    y: u32,<|>
}

// AFTER
#[derive()]
struct Point {
    x: u32,
    y: u32,
}
```

## `add_explicit_type`

Specify type for a let binding

```rust
// BEFORE
fn main() {
    let x<|> = 92;
}

// AFTER
fn main() {
    let x: i32 = 92;
}
```

## `add_impl`

Adds a new inherent impl for a type

```rust
// BEFORE
struct Ctx<T: Clone> {
     data: T,<|>
}

// AFTER
struct Ctx<T: Clone> {
     data: T,
}

impl<T: Clone> Ctx<T> {

}
```

## `add_impl_default_members`

Adds scaffold for overriding default impl members

```rust
// BEFORE
trait T {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl T for () {
    Type X = ();
    fn foo(&self) {}<|>

}

// AFTER
trait T {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl T for () {
    Type X = ();
    fn foo(&self) {}
    fn bar(&self) {}

}
```

## `add_impl_missing_members`

Adds scaffold for required impl members

```rust
// BEFORE
trait T {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl T for () {<|>

}

// AFTER
trait T {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl T for () {
    fn foo(&self) { unimplemented!() }

}
```

## `apply_demorgan`

Apply [De Morgan's law](https://en.wikipedia.org/wiki/De_Morgan%27s_laws).
This transforms expressions of the form `!l || !r` into `!(l && r)`.
This also works with `&&`. This assist can only be applied with the cursor
on either `||` or `&&`, with both operands being a negation of some kind.
This means something of the form `!x` or `x != y`.

```rust
// BEFORE
fn main() {
    if x != 4 ||<|> !y {}
}

// AFTER
fn main() {
    if !(x == 4 && y) {}
}
```

## `change_visibility`

Adds or changes existing visibility specifier.

```rust
// BEFORE
fn<|> frobnicate() {}

// AFTER
pub(crate) fn frobnicate() {}
```

## `convert_to_guarded_return`

Replace a large conditional with a guarded return.

```rust
// BEFORE
fn main() {
    <|>if cond {
        foo();
        bar();
    }
}

// AFTER
fn main() {
    if !cond {
        return;
    }
    foo();
    bar();
}
```

## `fill_match_arms`

Adds missing clauses to a `match` expression.

```rust
// BEFORE
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        <|>
    }
}

// AFTER
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move{ distance } => (),
        Action::Stop => (),
    }
}
```
