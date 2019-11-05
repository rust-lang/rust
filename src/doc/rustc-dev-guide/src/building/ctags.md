# ctags

One of the challenges with rustc is that the RLS can't handle it, since it's a
bootstrapping compiler. This makes code navigation difficult. One solution is to
use `ctags`.

`ctags` has a long history and several variants. Exuberant Ctags seems to be
quite commonly distributed but it does not have out-of-box Rust support. Some
distributions seem to use [Universal Ctags][utags], which is a maintained fork
and does have built-in Rust support.

The following script can be used to set up Exuberant Ctags:
[https://github.com/nikomatsakis/rust-etags][etags].

`ctags` integrates into emacs and vim quite easily. The following can then be
used to build and generate tags:

```console
$ rust-ctags src/lib* && ./x.py build <something>
```

This allows you to do "jump-to-def" with whatever functions were around when
you last built, which is ridiculously useful.

[etags]: https://github.com/nikomatsakis/rust-etags
[utags]: https://github.com/universal-ctags/ctags
