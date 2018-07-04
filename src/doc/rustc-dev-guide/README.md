This is a collaborate effort to build a guide that explains how rustc
works. The aim of the guide is to help new contributors get oriented
to rustc, as well as to help more experienced folks in figuring out
some new part of the compiler that they haven't worked on before.

[You can read the latest version of the guide here.](https://rust-lang-nursery.github.io/rustc-guide/)

You may also find the rustdocs [for the compiler itself][rustdocs] useful.

[rustdocs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/

The guide can be useful today, but it has a lot of work still go.
Once it gets more complete, the plan is probably to move it into the
[main Rust repository](https://github.com/rust-lang/rust/).

### Contributing to the guide

If you'd like to help finish the guide, we'd love to have you! The
main tracking issue for the guide
[can be found here](https://github.com/rust-lang-nursery/rustc-guide/issues/6). From
there, you can find a list of all the planned chapters and subsections
-- if you think something is missing, please open an issue about it!
Otherwise, find a chapter that sounds interesting to you and then go
to its associated issue. There should be a list of things to do.

**In general, if you don't know how the compiler works, that is not a
problem!** In that case, what we will do is to schedule a bit of time
for you to talk with someone who **does** know the code, or who wants
to pair with you and figure it out.  Then you can work on writing up
what you learned.

To help prevent accidentally introducing broken links, we use the
`mdbook-linkcheck`. If installed on your machine `mdbook` will automatically
invoke this link checker, otherwise it will emit a warning saying it couldn't
be found.

```bash
> cargo install mdbook-linkcheck
```
You will need `mdbook` version `>= 0.1`. `linkcheck` will be run automatically
when you run `mdbook build`.
