Regarding https://github.com/rust-lang/rust/pull/159581#issuecomment-5095419394.

### Suspicious contexts (single -> list)

In all these cases if `$expr` has a top level attribute inside it, it will migrate from a single to list context if the parentheses from the macro variable are not correctly preserved in AST.

```rust
// method call arguments
r.method($expr, ...)

// array element or size
[$expr, 11, 12]

// function call argument
func($expr, 11, 12)

// tuple element
($expr, 11, 12)

// const generic arguments (if supported at all)
foo::<$expr>()
```

### Suspicious contexts (expr -> stmt)

In all these cases if `$expr` has a top level attribute inside it, it will migrate from a single expression to statement context if the parentheses from the macro variable are not correctly preserved in AST.

```rust
// $expr is:

// postfix await, yield or use
#[attr] val.await
#[attr] val.yield
#[attr] val.use

// field or method access
#[attr] val.field
#[attr] val.method()

// function call or indexing
#[attr] val()
#[attr] val[]

// expression
#[attr] val?

// path
#[attr] path
```

If an attribute starts a statement as a part of some larger expression (e.g. a binary operator), it will already be reported as an error by https://github.com/rust-lang/rust/pull/160235#issuecomment-5159173426.

### Conclusion

These examples are only ambiguous if any of the involved attributes are active (including `cfg`), except that `cfg_attr` is ok (because we know what it does, and it does not depend on context).
So we don't need to gate any of this during parsing, and can continue using expansion-time gating.
