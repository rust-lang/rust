% Associated Types

Associated types are a powerful part of Rust’s type system. They’re related to
the idea of a ‘type family’, in other words, grouping multiple types together. That
description is a bit abstract, so let’s dive right into an example. If you want
to write a `Graph` trait, you have two types to be generic over: the node type
and the edge type. So you might write a trait, `Graph<N, E>`, that looks like
this:

```rust
trait Graph<N, E> {
    fn has_edge(&self, &N, &N) -> bool;
    fn edges(&self, &N) -> Vec<E>;
    // Etc.
}
```

While this sort of works, it ends up being awkward. For example, any function
that wants to take a `Graph` as a parameter now _also_ needs to be generic over
the `N`ode and `E`dge types too:

```rust,ignore
fn distance<N, E, G: Graph<N, E>>(graph: &G, start: &N, end: &N) -> u32 { ... }
```

Our distance calculation works regardless of our `Edge` type, so the `E` stuff in
this signature is a distraction.

What we really want to say is that a certain `E`dge and `N`ode type come together
to form each kind of `Graph`. We can do that with associated types:

```rust
trait Graph {
    type N;
    type E;

    fn has_edge(&self, &Self::N, &Self::N) -> bool;
    fn edges(&self, &Self::N) -> Vec<Self::E>;
    // Etc.
}
```

Now, our clients can be abstract over a given `Graph`:

```rust,ignore
fn distance<G: Graph>(graph: &G, start: &G::N, end: &G::N) -> u32 { ... }
```

No need to deal with the `E`dge type here!

Let’s go over all this in more detail.

## Defining associated types

Let’s build that `Graph` trait. Here’s the definition:

```rust
trait Graph {
    type N;
    type E;

    fn has_edge(&self, &Self::N, &Self::N) -> bool;
    fn edges(&self, &Self::N) -> Vec<Self::E>;
}
```

Simple enough. Associated types use the `type` keyword, and go inside the body
of the trait, with the functions.

These type declarations work the same way as those for functions. For example,
if we wanted our `N` type to implement `Display`, so we can print the nodes out,
we could do this:

```rust
use std::fmt;

trait Graph {
    type N: fmt::Display;
    type E;

    fn has_edge(&self, &Self::N, &Self::N) -> bool;
    fn edges(&self, &Self::N) -> Vec<Self::E>;
}
```

## Implementing associated types

Just like any trait, traits that use associated types use the `impl` keyword to
provide implementations. Here’s a simple implementation of Graph:

```rust
# trait Graph {
#     type N;
#     type E;
#     fn has_edge(&self, &Self::N, &Self::N) -> bool;
#     fn edges(&self, &Self::N) -> Vec<Self::E>;
# }
struct Node;

struct Edge;

struct MyGraph;

impl Graph for MyGraph {
    type N = Node;
    type E = Edge;

    fn has_edge(&self, n1: &Node, n2: &Node) -> bool {
        true
    }

    fn edges(&self, n: &Node) -> Vec<Edge> {
        Vec::new()
    }
}
```

This silly implementation always returns `true` and an empty `Vec<Edge>`, but it
gives you an idea of how to implement this kind of thing. We first need three
`struct`s, one for the graph, one for the node, and one for the edge. If it made
more sense to use a different type, that would work as well, we’re going to
use `struct`s for all three here.

Next is the `impl` line, which is an implementation like any other trait.

From here, we use `=` to define our associated types. The name the trait uses
goes on the left of the `=`, and the concrete type we’re `impl`ementing this
for goes on the right. Finally, we use the concrete types in our function
declarations.

## Trait objects with associated types

There’s one more bit of syntax we should talk about: trait objects. If you
try to create a trait object from a trait with an associated type, like this:

```rust,ignore
# trait Graph {
#     type N;
#     type E;
#     fn has_edge(&self, &Self::N, &Self::N) -> bool;
#     fn edges(&self, &Self::N) -> Vec<Self::E>;
# }
# struct Node;
# struct Edge;
# struct MyGraph;
# impl Graph for MyGraph {
#     type N = Node;
#     type E = Edge;
#     fn has_edge(&self, n1: &Node, n2: &Node) -> bool {
#         true
#     }
#     fn edges(&self, n: &Node) -> Vec<Edge> {
#         Vec::new()
#     }
# }
let graph = MyGraph;
let obj = Box::new(graph) as Box<Graph>;
```

You’ll get two errors:

```text
error: the value of the associated type `E` (from the trait `main::Graph`) must
be specified [E0191]
let obj = Box::new(graph) as Box<Graph>;
          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
24:44 error: the value of the associated type `N` (from the trait
`main::Graph`) must be specified [E0191]
let obj = Box::new(graph) as Box<Graph>;
          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

We can’t create a trait object like this, because we don’t know the associated
types. Instead, we can write this:

```rust
# trait Graph {
#     type N;
#     type E;
#     fn has_edge(&self, &Self::N, &Self::N) -> bool;
#     fn edges(&self, &Self::N) -> Vec<Self::E>;
# }
# struct Node;
# struct Edge;
# struct MyGraph;
# impl Graph for MyGraph {
#     type N = Node;
#     type E = Edge;
#     fn has_edge(&self, n1: &Node, n2: &Node) -> bool {
#         true
#     }
#     fn edges(&self, n: &Node) -> Vec<Edge> {
#         Vec::new()
#     }
# }
let graph = MyGraph;
let obj = Box::new(graph) as Box<Graph<N=Node, E=Edge>>;
```

The `N=Node` syntax allows us to provide a concrete type, `Node`, for the `N`
type parameter. Same with `E=Edge`. If we didn’t provide this constraint, we
couldn’t be sure which `impl` to match this trait object to.
