% Using trait objects

> **[FIXME]** What are uses of trait objects other than heterogeneous collections?

Trait objects are useful primarily when _heterogeneous_ collections of objects
need to be treated uniformly; it is the closest that Rust comes to
object-oriented programming.

```rust
struct Frame  { ... }
struct Button { ... }
struct Label  { ... }

trait Widget  { ... }

impl Widget for Frame  { ... }
impl Widget for Button { ... }
impl Widget for Label  { ... }

impl Frame {
    fn new(contents: &[Box<Widget>]) -> Frame {
        ...
    }
}

fn make_gui() -> Box<Widget> {
    let b: Box<Widget> = box Button::new(...);
    let l: Box<Widget> = box Label::new(...);

    box Frame::new([b, l]) as Box<Widget>
}
```

By using trait objects, we can set up a GUI framework with a `Frame` widget that
contains a heterogeneous collection of children widgets.

**Pros**:

* _Heterogeneity_. When you need it, you really need it.
* _Code size_. Unlike generics, trait objects do not generate specialized
  (monomorphized) versions of code, which can greatly reduce code size.

**Cons**:

* _No generic methods_. Trait objects cannot currently provide generic methods.
* _Dynamic dispatch and fat pointers_. Trait objects inherently involve
  indirection and vtable dispatch, which can carry a performance penalty.
* _No Self_. Except for the method receiver argument, methods on trait objects
  cannot use the `Self` type.
