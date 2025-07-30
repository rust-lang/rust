# `codegen-source-order`

---

This feature allows you to have a predictive and
deterministic order for items after codegen, which
is the same as in source code.

For every `CodegenUnit`, local `MonoItem`s would
be sorted by `(Span, SymbolName)`, which
makes codegen tests rely on the order of items in
source files work.
