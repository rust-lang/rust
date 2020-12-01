Private modules cannot be publicly re-exported. This error indicates that you
attempted to `pub use` a module that was not itself public.

Erroneous code example:

```compile_fail,E0365
mod foo {
    pub const X: u32 = 1;
}

pub use foo as foo2;

fn main() {}
```

The solution to this problem is to ensure that the module that you are
re-exporting is itself marked with `pub`:

```
pub mod foo {
    pub const X: u32 = 1;
}

pub use foo as foo2;

fn main() {}
```

See the [Use Declarations][use-declarations] section of the reference for
more information on this topic.

[use-declarations]: https://doc.rust-lang.org/reference/items/use-declarations.html
