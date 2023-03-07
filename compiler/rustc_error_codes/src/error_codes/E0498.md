The `plugin` attribute was malformed.

Erroneous code example:

```compile_fail,E0498
#![feature(plugin)]
#![plugin(foo(args))] // error: invalid argument
#![plugin(bar="test")] // error: invalid argument
```

The `#[plugin]` attribute should take a single argument: the name of the plugin.

For example, for the plugin `foo`:

```ignore (requires external plugin crate)
#![feature(plugin)]
#![plugin(foo)] // ok!
```

See the [`plugin` feature] section of the Unstable book for more details.

[`plugin` feature]: https://doc.rust-lang.org/nightly/unstable-book/language-features/plugin.html
