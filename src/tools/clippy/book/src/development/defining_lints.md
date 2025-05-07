# Define New Lints

The first step in the journey of a new lint is the definition
and registration of the lint in Clippy's codebase.
We can use the Clippy dev tools to handle this step since setting up the
lint involves some boilerplate code.

#### Lint types

A lint type is the category of items and expressions in which your lint focuses on.

As of the writing of this documentation update, there are 11 _types_ of lints
besides the numerous standalone lints living under `clippy_lints/src/`:

- `cargo`
- `casts`
- `functions`
- `loops`
- `matches`
- `methods`
- `misc_early`
- `operators`
- `transmute`
- `types`
- `unit_types`

These types group together lints that share some common behaviors. For instance,
`functions` groups together lints that deal with some aspects of functions in
Rust, like definitions, signatures and attributes.

For more information, feel free to compare the lint files under any category
with [All Clippy lints][all_lints] or ask one of the maintainers.

## Lint name

A good lint name is important, make sure to check the [lint naming
guidelines][lint_naming]. Don't worry, if the lint name doesn't fit, a Clippy
team member will alert you in the PR process.

---

We'll name our example lint that detects functions named "foo" `foo_functions`.
Check the [lint naming guidelines][lint_naming] to see why this name makes
sense.

## Add and Register the Lint

Now that a name is chosen, we shall register `foo_functions` as a lint to the
codebase. There are two ways to register a lint.

### Standalone

If you believe that this new lint is a standalone lint (that doesn't belong to
any specific [type](#lint-types) like `functions` or `loops`), you can run the
following command in your Clippy project:

```sh
$ cargo dev new_lint --name=lint_name --pass=late --category=pedantic
```

There are two things to note here:

1. `--pass`: We set `--pass=late` in this command to do a late lint pass. The
   alternative is an `early` lint pass. We will discuss this difference in the
   [Lint Passes] chapter.
2. `--category`: If not provided, the `category` of this new lint will default
   to `nursery`.

The `cargo dev new_lint` command will create a new file:
`clippy_lints/src/foo_functions.rs` as well as [register the
lint](#lint-registration).

Overall, you should notice that the following files are modified or created:

```sh
$ git status
On branch foo_functions
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   CHANGELOG.md
	modified:   clippy_lints/src/lib.register_lints.rs
	modified:   clippy_lints/src/lib.register_pedantic.rs
	modified:   clippy_lints/src/lib.rs

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	clippy_lints/src/foo_functions.rs
	tests/ui/foo_functions.rs
```


### Specific Type

> **Note**: Lint types are listed in the ["Lint types"](#lint-types) section

If you believe that this new lint belongs to a specific type of lints,
you can run `cargo dev new_lint` with a `--type` option.

Since our `foo_functions` lint is related to function calls, one could
argue that we should put it into a group of lints that detect some behaviors
of functions, we can put it in the `functions` group.

Let's run the following command in your Clippy project:

```sh
$ cargo dev new_lint --name=foo_functions --type=functions --category=pedantic
```

This command will create, among other things, a new file:
`clippy_lints/src/{type}/foo_functions.rs`.
In our case, the path will be `clippy_lints/src/functions/foo_functions.rs`.

Notice how this command has a `--type` flag instead of `--pass`. Unlike a standalone
definition, this lint won't be registered in the traditional sense. Instead, you will
call your lint from within the type's lint pass, found in `clippy_lints/src/{type}/mod.rs`.

A _type_ is just the name of a directory in `clippy_lints/src`, like `functions` in
the example command. Clippy groups together some lints that share common behaviors,
so if your lint falls into one, it would be best to add it to that type.

Overall, you should notice that the following files are modified or created:

```sh
$ git status
On branch foo_functions
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   CHANGELOG.md
	modified:   clippy_lints/src/declared_lints.rs
	modified:   clippy_lints/src/functions/mod.rs

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	clippy_lints/src/functions/foo_functions.rs
	tests/ui/foo_functions.rs
```


## The `declare_clippy_lint` macro

After `cargo dev new_lint`, you should see a macro with the name
`declare_clippy_lint`. It will be in the same file if you defined a standalone
lint, and it will be in `mod.rs` if you defined a type-specific lint.

The macro looks something like this:

```rust
declare_clippy_lint! {
    /// ### What it does
    ///
    /// // Describe here what does the lint do.
    ///
    /// Triggers when detects...
    ///
    /// ### Why is this bad?
    ///
    /// // Describe why this pattern would be bad
    ///
    /// It can lead to...
    ///
    /// ### Example
    /// ```rust
    /// // example code where Clippy issues a warning
    /// ```
    /// Use instead:
    /// ```rust
    /// // example code which does not raise Clippy warning
    /// ```
    #[clippy::version = "1.70.0"] // <- In which version was this implemented, keep it up to date!
    pub LINT_NAME, // <- The lint name IN_ALL_CAPS
    pedantic, // <- The lint group
    "default lint description" // <- A lint description, e.g. "A function has an unit return type."
}
```

## Lint registration

If we run the `cargo dev new_lint` command for a new lint, the lint will be
automatically registered and there is nothing more to do.

However, sometimes we might want to declare a new lint by hand. In this case,
we'd use `cargo dev update_lints` command afterwards.

When a lint is manually declared, we might need to register the lint pass
manually in the `register_lints` function in `clippy_lints/src/lib.rs`:

```rust
store.register_late_pass(|_| Box::new(foo_functions::FooFunctions));
```

As you might have guessed, where there's something late, there is something
early: in Clippy there is a `register_early_pass` method as well. More on early
vs. late passes in the [Lint Passes] chapter.

Without a call to one of `register_early_pass` or `register_late_pass`, the lint
pass in question will not be run.


[all_lints]: https://rust-lang.github.io/rust-clippy/master/
[lint_naming]: https://rust-lang.github.io/rfcs/0344-conventions-galore.html#lints
[Lint Passes]: lint_passes.md
