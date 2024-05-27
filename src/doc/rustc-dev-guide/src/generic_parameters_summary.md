# Generic parameter definitions

This chapter will discuss how rustc tracks what generic parameters are introduced by an item. For example given some struct defined via `struct Foo<T>` how does rustc track that `Foo` defines some type parameter `T` and nothing else?

This will *not* cover how we track generic parameters introduced via `for<'a>` syntax (i.e. in where clauses or `fn` types), which is covered elsewhere in the [chapter on `Binder`s ][ch_binders].

[ch_binders]: ./ty_module/binders.md