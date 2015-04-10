- Feature Name: variadic-generics
- Start Date: 2015-04-03
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary


# Motivation

Why are we doing this? What use cases does it support? What is the expected outcome?

# Detailed design

## Introduce the `Tuple` trait

The `Tuple` trait is implemented for tuples of any arity. It is a lang
item.

```rust
trait Tuple { }
```

## Expandable parameters

In a `fn` signature, a `..` may appear before the type of any
argument. This type must implement the `Tuple` trait. The `..`
indicates that the single 

```rust
trait Fn<A:Tuple>: FnMut<A> {
    fn call(&self, args: ..A) -> Self::Output;
}
```

# Drawbacks

Why should we *not* do this?

# Alternatives

What other designs have been considered? What is the impact of not doing this?

# Unresolved questions

What parts of the design are still TBD?
