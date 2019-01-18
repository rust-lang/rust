This is a collaborative effort to build a guide that explains how rustc
works. The aim of the guide is to help new contributors get oriented
to rustc, as well as to help more experienced folks in figuring out
some new part of the compiler that they haven't worked on before.

[You can read the latest version of the guide here.](https://rust-lang-nursery.github.io/rustc-guide/)

You may also find the rustdocs [for the compiler itself][rustdocs] useful.

[rustdocs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/

### Contributing to the guide

The guide is useful today, but it has a lot of work still go.

If you'd like to help improve the guide, we'd love to have you! You can find
plenty of issues on the [issue
tracker](https://github.com/rust-lang/rustc-guide/issues). Just post a
comment on the issue you would like to work on to make sure that we don't
accidentally duplicate work. If you think something is missing, please open an
issue about it!

**In general, if you don't know how the compiler works, that is not a
problem!** In that case, what we will do is to schedule a bit of time
for you to talk with someone who **does** know the code, or who wants
to pair with you and figure it out.  Then you can work on writing up
what you learned.

In general, when writing about a particular part of the compiler's code, we
recommend that you link to the relevant parts of the [rustc
rustdocs][rustdocs].

To help prevent accidentally introducing broken links, we use the
`mdbook-linkcheck`. If installed on your machine `mdbook` will automatically
invoke this link checker, otherwise it will emit a warning saying it couldn't
be found.

```bash
> cargo install mdbook-linkcheck
```

You will need `mdbook` version `>= 0.2`. `linkcheck` will be run automatically
when you run `mdbook build`.
