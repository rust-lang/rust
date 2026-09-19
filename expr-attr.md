Attributes in expressions, purely from syntactic point of view, at parsing time.
I.e. no distinction between active or inert, or built-in or custom attributes.
And without considering https://github.com/rust-lang/rust/pull/159581#issuecomment-5095419394.

### Not syntactically suspicious (top level expressions):

Definitely can stabilize after accounting for macro attributes and https://github.com/rust-lang/rust/pull/159581#issuecomment-5095419394.

```rust
// let initializer
let x = #[attr] 10;
let x = #[attr] 10 else { ... };

// const or static item initializer
const C: u8 = #[attr] 10;
static C: u8 = #[attr] 10;

// enum discriminant value
enum E { V = #[attr] 10 }

// field default
struct S { field: u8 = #[attr] 10 }

// key value attribute
#[key = #[attr] value]

// `match` arm
match 10 {
    11 => #[attr] 12,
}

// struct literals fields
let x = Struct { field: #[attr] 10, ... }

// struct literals rest
let x = Struct { field: 10, ..#[attr] base }

// for loop iterator
for x in #[attr] 10 { ... }

// if condition
if #[attr] 10 { ... }

// while condition
while #[attr] 10 { ... }

// match scrutinee
match #[attr] 10 { ... }

// method call arguments
r.method(#[attr] 10, ...)

// array element or size
[#[attr] 10, 11, 12]
[#[attr] 10; 11]
[10; #[attr] 11]
[u8; #[attr] 11]

// function call argument
func(#[attr] 10, 11, 12)

// tuple element
(#[attr] 10, 11, 12)

// move expression
move(#[attr] 10)

// indexing argument
array[#[attr] 10]

// parentheses
(#[attr] 10)

// const generic defaults
fn foo<const C: u8 = #[attr] 10>() {}

// const parameter constraint
foo::<C = #[attr] 10>()

// const generic arguments (if supported at all)
foo::<#[attr] 10>()

// type ascription
builtin # type_ascribe(#[attr] 10, u8)

// unsafe binder casts
builtin # wrap_binder(#[attr] 10, u8)

// various inline asm operands
asm!("code", in("r") #[attr] x)

// some syntax for contracts, I didn't look closely
```

### Semi-suspicious (top level expressions):

I don't see any actual issues, just would be interesting how often these will occur in the crater run.

Can stabilize (after accounting for macro attributes and https://github.com/rust-lang/rust/pull/159581#issuecomment-5095419394) or not stabilize.

```rust
// closure body
let x = || #[attr] 10;

// break, return, yield, become, and yeet expressions
break #[attr] 10
return #[attr] 10
yield #[attr] 10
become #[attr] 10
do yeet #[attr] 10

// match or pattern guard conditions
pat if #[attr] true

// `expr` matcher as an await or yield receiver
$expr.await // `$expr` is `#[attr] 10`
$expr.yield // `$expr` is `#[attr] 10`

// `expr` matcher as a use receiver
$expr.use // `$expr` is `#[attr] 10`

// `expr` matcher as a field receiver
$expr.field // `$expr` is `#[attr] 10`

// `expr` matcher as a method receiver
$expr.method() // `$expr` is `#[attr] 10`

// `expr` matcher as a callable in a function call
$expr() // `$expr` is `#[attr] 10`

// `expr` matcher as an indexable expression
$expr[] // `$expr` is `#[attr] 10`

// `expr` matcher as a try expression
$expr? // `$expr` is `#[attr] 10`
```

### Semi-suspicious (non-top level expressions):

I don't see any actual issues, just would be interesting how often these will occur in the crater run.

Can stabilize (after accounting for macro attributes and https://github.com/rust-lang/rust/pull/159581#issuecomment-5095419394) or not stabilize.

```rust
// binary operator rhs
10 + #[attr] 11

// assignment rhs
lhs = #[attr] rhs
lhs += #[attr] rhs

// range rhs
10 .. #[attr] 11

// let expressions in if/while
if let x = #[attr] 10 && x == 11 { ... }

// prefix unary operator
- #[attr] 10

// reference operator
& #[attr] 10
& raw mut #[attr] 10
```

### Suspicious (top level expressions):

Cannot stabilize, this is a long stanging issue with too permissive pattern parsing.

```rust
// `expr` matcher as a pattern
match 10 { $expr => {} } // `$expr` is `#[attr] 10`

// `expr` matcher in a range pattern
match 10 { $expr .. $expr => {} } // `$expr` is `#[attr] 10`
```

### Suspicious (non-top level expressions):

Cannot stabilize.

```rust
// binary operator lhs
#[attr] 10 + 11

// assignment lhs
#[attr] lhs = rhs
#[attr] lhs += rhs

// range lhs
#[attr] 10 .. 11

// cast expressions
#[attr] 10 as u8
```
