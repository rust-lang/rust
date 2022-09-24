# Guiding principles and rationale

When deciding on style guidelines, the style team tried to be guided by the
following principles (in rough priority order):

* readability
    - scan-ability
    - avoiding misleading formatting
    - accessibility - readable and editable by users using the the widest
      variety of hardware, including non-visual accessibility interfaces
    - readability of code when quoted in rustc error messages

* aesthetics
    - sense of 'beauty'
    - consistent with other languages/tools

* specifics
    - compatibility with version control practices - preserving diffs,
      merge-friendliness, etc.
    - preventing right-ward drift
    - minimising vertical space

* application
    - ease of manual application
    - ease of implementation (in Rustfmt, and in other tools/editors/code generators)
    - internal consistency
    - simplicity of formatting rules


## Overarching guidelines

Prefer block indent over visual indent. E.g.,

```rust
// Block indent
a_function_call(
    foo,
    bar,
);

// Visual indent
a_function_call(foo,
                bar);
```

This makes for smaller diffs (e.g., if `a_function_call` is renamed in the above
example) and less rightward drift.

Lists should have a trailing comma when followed by a newline, see the block
indent example above. This choice makes moving code (e.g., by copy and paste)
easier and makes smaller diffs.
