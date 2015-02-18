% Using `unsafe`

### Unconditionally guarantee safety, or mark API as `unsafe`. **[FIXME: needs RFC]**

Memory safety, type safety, and data race freedom are basic assumptions for all
Rust code.

APIs that use `unsafe` blocks internally thus have two choices:

* They can guarantee safety _unconditionally_ (i.e., regardless of client
  behavior or inputs) and be exported as safe code. Any safety violation is then
  the library's fault, not the client's fault.

* They can export potentially unsafe functions with the `unsafe` qualifier. In
  this case, the documentation should make very clear the conditions under which
  safety is guaranteed.

The result is that a client program can never violate safety merely by having a
bug; it must have explicitly opted out by using an `unsafe` block.

Of the two options for using `unsafe`, creating such safe abstractions (the
first option above) is strongly preferred.
