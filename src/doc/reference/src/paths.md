# Paths

A _path_ is a sequence of one or more path components _logically_ separated by
a namespace qualifier (`::`). If a path consists of only one component, it may
refer to either an [item] or a [variable] in a local control
scope. If a path has multiple components, it refers to an item.

[item]: items.html
[variable]: variables.html

Every item has a _canonical path_ within its crate, but the path naming an item
is only meaningful within a given crate. There is no global namespace across
crates; an item's canonical path merely identifies it within the crate.

Two examples of simple paths consisting of only identifier components:

```{.ignore}
x;
x::y::z;
```

Path components are usually [identifiers], but they may
also include angle-bracket-enclosed lists of type arguments. In
[expression] context, the type argument list is given
after a `::` namespace qualifier in order to disambiguate it from a
relational expression involving the less-than symbol (`<`). In type
expression context, the final namespace qualifier is omitted.

[identifiers]: identifiers.html
[expression]: expressions.html

Two examples of paths with type arguments:

```rust
# struct HashMap<K, V>(K,V);
# fn f() {
# fn id<T>(t: T) -> T { t }
type T = HashMap<i32,String>; // Type arguments used in a type expression
let x  = id::<i32>(10);       // Type arguments used in a call expression
# }
```

Paths can be denoted with various leading qualifiers to change the meaning of
how it is resolved:

* Paths starting with `::` are considered to be global paths where the
  components of the path start being resolved from the crate root. Each
  identifier in the path must resolve to an item.

```rust
mod a {
    pub fn foo() {}
}
mod b {
    pub fn foo() {
        ::a::foo(); // call a's foo function
    }
}
# fn main() {}
```

* Paths starting with the keyword `super` begin resolution relative to the
  parent module. Each further identifier must resolve to an item.

```rust
mod a {
    pub fn foo() {}
}
mod b {
    pub fn foo() {
        super::a::foo(); // call a's foo function
    }
}
# fn main() {}
```

* Paths starting with the keyword `self` begin resolution relative to the
  current module. Each further identifier must resolve to an item.

```rust
fn foo() {}
fn bar() {
    self::foo();
}
# fn main() {}
```

Additionally keyword `super` may be repeated several times after the first
`super` or `self` to refer to ancestor modules.

```rust
mod a {
    fn foo() {}

    mod b {
        mod c {
            fn foo() {
                super::super::foo(); // call a's foo function
                self::super::super::foo(); // call a's foo function
            }
        }
    }
}
# fn main() {}
```
