% Loops

### Prefer `for` to `while`. [FIXME: needs RFC]

A `for` loop is preferable to a `while` loop, unless the loop counts in a
non-uniform way (making it difficult to express using `for`).

### Guidelines for `loop`. [FIXME]

> **[FIXME]** When is `loop` recommended? Some possibilities:
> * For optimistic retry algorithms
> * For servers
> * To avoid mutating local variables sometimes needed to fit `while`
