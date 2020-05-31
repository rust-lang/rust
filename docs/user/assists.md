# Assists

Cursor position or selection is signified by `┃` character.


## `add_custom_impl`

Adds impl block for derived trait.

```rust
// BEFORE
#[derive(Deb┃ug, Display)]
struct S;

// AFTER
#[derive(Display)]
struct S;

impl Debug for S {
    $0
}
```

## `add_derive`

Adds a new `#[derive()]` clause to a struct or enum.

```rust
// BEFORE
struct Point {
    x: u32,
    y: u32,┃
}

// AFTER
#[derive($0)]
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

## `add_from_impl_for_enum`

Adds a From impl for an enum variant with one tuple field.

```rust
// BEFORE
enum A { ┃One(u32) }

// AFTER
enum A { One(u32) }

impl From<u32> for A {
    fn from(v: u32) -> Self {
        A::One(v)
    }
}
```

## `add_function`

Adds a stub function with a signature matching the function under the cursor.

```rust
// BEFORE
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    bar┃("", baz());
}


// AFTER
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    bar("", baz());
}

fn bar(arg: &str, baz: Baz) {
    ${0:todo!()}
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
    $0
}
```

## `add_impl_default_members`

Adds scaffold for overriding default impl members.

```rust
// BEFORE
trait Trait {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl Trait for () {
    Type X = ();
    fn foo(&self) {}┃

}

// AFTER
trait Trait {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl Trait for () {
    Type X = ();
    fn foo(&self) {}
    $0fn bar(&self) {}

}
```

## `add_impl_missing_members`

Adds scaffold for required impl members.

```rust
// BEFORE
trait Trait<T> {
    Type X;
    fn foo(&self) -> T;
    fn bar(&self) {}
}

impl Trait<u32> for () {┃

}

// AFTER
trait Trait<T> {
    Type X;
    fn foo(&self) -> T;
    fn bar(&self) {}
}

impl Trait<u32> for () {
    fn foo(&self) -> u32 {
        ${0:todo!()}
    }

}
```

## `add_new`

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
    fn $0new(data: T) -> Self { Self { data } }
}

```

## `add_turbo_fish`

Adds `::<_>` to a call of a generic method or function.

```rust
// BEFORE
fn make<T>() -> T { todo!() }
fn main() {
    let x = make┃();
}

// AFTER
fn make<T>() -> T { todo!() }
fn main() {
    let x = make::<${0:_}>();
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
    if x != 4 ||┃ !y {}
}

// AFTER
fn main() {
    if !(x == 4 && y) {}
}
```

## `auto_import`

If the name is unresolved, provides all possible imports for it.

```rust
// BEFORE
fn main() {
    let map = HashMap┃::new();
}

// AFTER
use std::collections::HashMap;

fn main() {
    let map = HashMap::new();
}
```

## `change_lifetime_anon_to_named`

Change an anonymous lifetime to a named lifetime.

```rust
// BEFORE
impl Cursor<'_┃> {
    fn node(self) -> &SyntaxNode {
        match self {
            Cursor::Replace(node) | Cursor::Before(node) => node,
        }
    }
}

// AFTER
impl<'a> Cursor<'a> {
    fn node(self) -> &SyntaxNode {
        match self {
            Cursor::Replace(node) | Cursor::Before(node) => node,
        }
    }
}
```

## `change_return_type_to_result`

Change the function's return type to Result.

```rust
// BEFORE
fn foo() -> i32┃ { 42i32 }

// AFTER
fn foo() -> Result<i32, ${0:_}> { Ok(42i32) }
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
        $0Action::Move { distance } => {}
        Action::Stop => {}
    }
}
```

## `fix_visibility`

Makes inaccessible item public.

```rust
// BEFORE
mod m {
    fn frobnicate() {}
}
fn main() {
    m::frobnicate┃() {}
}

// AFTER
mod m {
    $0pub(crate) fn frobnicate() {}
}
fn main() {
    m::frobnicate() {}
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
    let $0var_name = (1 + 2);
    var_name * 4;
}
```

## `invert_if`

Apply invert_if
This transforms if expressions of the form `if !x {A} else {B}` into `if x {B} else {A}`
This also works with `!=`. This assist can only be applied with the cursor
on `if`.

```rust
// BEFORE
fn main() {
    if┃ !y { A } else { B }
}

// AFTER
fn main() {
    if y { B } else { A }
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

## `merge_imports`

Merges two imports with a common prefix.

```rust
// BEFORE
use std::┃fmt::Formatter;
use std::io;

// AFTER
use std::{fmt::Formatter, io};
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

## `remove_mut`

Removes the `mut` keyword.

```rust
// BEFORE
impl Walrus {
    fn feed(&mut┃ self, amount: u32) {}
}

// AFTER
impl Walrus {
    fn feed(&self, amount: u32) {}
}
```

## `reorder_fields`

Reorder the fields of record literals and record patterns in the same order as in
the definition.

```rust
// BEFORE
struct Foo {foo: i32, bar: i32};
const test: Foo = ┃Foo {bar: 0, foo: 1}

// AFTER
struct Foo {foo: i32, bar: i32};
const test: Foo = Foo {foo: 1, bar: 0}
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

## `replace_let_with_if_let`

Replaces `let` with an `if-let`.

```rust
// BEFORE

fn main(action: Action) {
    ┃let x = compute();
}

fn compute() -> Option<i32> { None }

// AFTER

fn main(action: Action) {
    if let Some(x) = compute() {
    }
}

fn compute() -> Option<i32> { None }
```

## `replace_qualified_name_with_use`

Adds a use statement for a given fully-qualified name.

```rust
// BEFORE
fn process(map: std::collections::┃HashMap<String, String>) {}

// AFTER
use std::collections::HashMap;

fn process(map: HashMap<String, String>) {}
```

## `replace_unwrap_with_match`

Replaces `unwrap` a `match` expression. Works for Result and Option.

```rust
// BEFORE
enum Result<T, E> { Ok(T), Err(E) }
fn main() {
    let x: Result<i32, i32> = Result::Ok(92);
    let y = x.┃unwrap();
}

// AFTER
enum Result<T, E> { Ok(T), Err(E) }
fn main() {
    let x: Result<i32, i32> = Result::Ok(92);
    let y = match x {
        Ok(a) => a,
        $0_ => unreachable!(),
    };
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

## `unwrap_block`

This assist removes if...else, for, while and loop control statements to just keep the body.

```rust
// BEFORE
fn foo() {
    if true {┃
        println!("foo");
    }
}

// AFTER
fn foo() {
    println!("foo");
}
```
