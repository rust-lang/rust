Private items cannot be publicly re-exported. This error indicates that you
attempted to `pub use` a type or value that was not itself public.

Erroneous code example:

```compile_fail,E0364
mod a {
    fn foo() {}

    mod a {
        pub use super::foo; // error!
    }
}
```

The solution to this problem is to ensure that the items that you are
re-exporting are themselves marked with `pub`:

```
mod a {
    pub fn foo() {} // ok!

    mod a {
        pub use super::foo;
    }
}
```

See the [Use Declarations][use-declarations] section of the reference for
more information on this topic.

[use-declarations]: https://doc.rust-lang.org/reference/items/use-declarations.html
