# Assists

Cursor position or selection is signified by `┃` character.


## `add_derive`

Adds a new `#[derive()]` clause to a struct or enum.

```rust
// BEFORE
struct Point {
    x: u32,
    y: u32,┃
}

// AFTER
#[derive()]
struct Point {
    x: u32,
    y: u32,
}
```

## `add_explicit_type`

Specify type for a let binding.

```rust
// BEFORE
fn main() {
    let x┃ = 92;
}

// AFTER
fn main() {
    let x: i32 = 92;
}
```

## `add_hash`

Adds a hash to a raw string literal.

```rust
// BEFORE
fn main() {
    r#"Hello,┃ World!"#;
}

// AFTER
fn main() {
    r##"Hello, World!"##;
}
```

## `add_impl`

Adds a new inherent impl for a type.

```rust
// BEFORE
struct Ctx<T: Clone> {
     data: T,┃
}

// AFTER
struct Ctx<T: Clone> {
     data: T,
}

impl<T: Clone> Ctx<T> {

}
```

## `add_impl_default_members`

Adds scaffold for overriding default impl members.

```rust
// BEFORE
trait T {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl T for () {
    Type X = ();
    fn foo(&self) {}┃

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

Adds scaffold for required impl members.

```rust
// BEFORE
trait T {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl T for () {┃

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

## `add_import`

Adds a use statement for a given fully-qualified path.

```rust
// BEFORE
fn process(map: std::collections::┃HashMap<String, String>) {}

// AFTER
use std::collections::HashMap;

fn process(map: HashMap<String, String>) {}
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
    if x != 4 ||┃ !y {}
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
┃fn frobnicate() {}

// AFTER
pub(crate) fn frobnicate() {}
```

## `convert_to_guarded_return`

Replace a large conditional with a guarded return.

```rust
// BEFORE
fn main() {
    ┃if cond {
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
        ┃
    }
}

// AFTER
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move { distance } => (),
        Action::Stop => (),
    }
}
```

## `flip_binexpr`

Flips operands of a binary expression.

```rust
// BEFORE
fn main() {
    let _ = 90 +┃ 2;
}

// AFTER
fn main() {
    let _ = 2 + 90;
}
```

## `flip_comma`

Flips two comma-separated items.

```rust
// BEFORE
fn main() {
    ((1, 2),┃ (3, 4));
}

// AFTER
fn main() {
    ((3, 4), (1, 2));
}
```

## `flip_trait_bound`

Flips two trait bounds.

```rust
// BEFORE
fn foo<T: Clone +┃ Copy>() { }

// AFTER
fn foo<T: Copy + Clone>() { }
```

## `inline_local_variable`

Inlines local variable.

```rust
// BEFORE
fn main() {
    let x┃ = 1 + 2;
    x * 4;
}

// AFTER
fn main() {
    (1 + 2) * 4;
}
```

## `introduce_variable`

Extracts subexpression into a variable.

```rust
// BEFORE
fn main() {
    ┃(1 + 2)┃ * 4;
}

// AFTER
fn main() {
    let var_name = (1 + 2);
    var_name * 4;
}
```

## `make_raw_string`

Adds `r#` to a plain string literal.

```rust
// BEFORE
fn main() {
    "Hello,┃ World!";
}

// AFTER
fn main() {
    r#"Hello, World!"#;
}
```

## `make_usual_string`

Turns a raw string into a plain string.

```rust
// BEFORE
fn main() {
    r#"Hello,┃ "World!""#;
}

// AFTER
fn main() {
    "Hello, \"World!\"";
}
```

## `merge_match_arms`

Merges identical match arms.

```rust
// BEFORE
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        ┃Action::Move(..) => foo(),
        Action::Stop => foo(),
    }
}

// AFTER
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move(..) | Action::Stop => foo(),
    }
}
```

## `move_arm_cond_to_match_guard`

Moves if expression from match arm body into a guard.

```rust
// BEFORE
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move { distance } => ┃if distance > 10 { foo() },
        _ => (),
    }
}

// AFTER
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move { distance } if distance > 10 => foo(),
        _ => (),
    }
}
```

## `move_bounds_to_where_clause`

Moves inline type bounds to a where clause.

```rust
// BEFORE
fn apply<T, U, ┃F: FnOnce(T) -> U>(f: F, x: T) -> U {
    f(x)
}

// AFTER
fn apply<T, U, F>(f: F, x: T) -> U where F: FnOnce(T) -> U {
    f(x)
}
```

## `move_guard_to_arm_body`

Moves match guard into match arm body.

```rust
// BEFORE
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move { distance } ┃if distance > 10 => foo(),
        _ => (),
    }
}

// AFTER
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move { distance } => if distance > 10 { foo() },
        _ => (),
    }
}
```

## `remove_dbg`

Removes `dbg!()` macro call.

```rust
// BEFORE
fn main() {
    ┃dbg!(92);
}

// AFTER
fn main() {
    92;
}
```

## `remove_hash`

Removes a hash from a raw string literal.

```rust
// BEFORE
fn main() {
    r#"Hello,┃ World!"#;
}

// AFTER
fn main() {
    r"Hello, World!";
}
```

## `replace_if_let_with_match`

Replaces `if let` with an else branch with a `match` expression.

```rust
// BEFORE
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    ┃if let Action::Move { distance } = action {
        foo(distance)
    } else {
        bar()
    }
}

// AFTER
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move { distance } => foo(distance),
        _ => bar(),
    }
}
```

## `split_import`

Wraps the tail of import into braces.

```rust
// BEFORE
use std::┃collections::HashMap;

// AFTER
use std::{collections::HashMap};
```
