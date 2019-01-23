This file offers some tips on the coding conventions for rustc.  This
chapter covers [formatting](#formatting), [coding for correctness](#cc),
[using crates from crates.io](#cio), and some tips on
[structuring your PR for easy review](#er).

<a name="formatting"></a>

# Formatting and the tidy script

rustc is slowly moving towards the [Rust standard coding style][fmt];
at the moment, however, it follows a rather more *chaotic* style.  We
do have some mandatory formatting conventions, which are automatically
enforced by a script we affectionately call the "tidy" script.  The
tidy script runs automatically when you do `./x.py test` and can be run
in isolation with `./x.py test src/tools/tidy`.

[fmt]: https://github.com/rust-lang-nursery/fmt-rfcs

<a name="copyright"></a>

### Copyright notice

In the past, files begin with a copyright and license notice. Please **omit**
this notice for new files licensed under the standard terms (dual
MIT/Apache-2.0).

All of the copyright notices should be gone by now, but if you come across one
in the rust-lang/rust repo, feel free to open a PR to remove it.

## Line length

Lines should be at most 100 characters. It's even better if you can
keep things to 80.

**Ignoring the line length limit.** Sometimes – in particular for
tests – it can be necessary to exempt yourself from this limit. In
that case, you can add a comment towards the top of the file (after
the copyright notice) like so:

```rust
// ignore-tidy-linelength
```

## Tabs vs spaces

Prefer 4-space indent.

<a name="cc"></a>

# Coding for correctness

Beyond formatting, there are a few other tips that are worth
following.

## Prefer exhaustive matches

Using `_` in a match is convenient, but it means that when new
variants are added to the enum, they may not get handled correctly.
Ask yourself: if a new variant were added to this enum, what's the
chance that it would want to use the `_` code, versus having some
other treatment?  Unless the answer is "low", then prefer an
exhaustive match. (The same advice applies to `if let` and `while
let`, which are effectively tests for a single variant.)

## Use "TODO" comments for things you don't want to forget

As a useful tool to yourself, you can insert a `// TODO` comment
for something that you want to get back to before you land your PR:

```rust,ignore
fn do_something() {
    if something_else {
        unimplemented!(); // TODO write this
    }
}
```

The tidy script will report an error for a `// TODO` comment, so this
code would not be able to land until the TODO is fixed (or removed).

This can also be useful in a PR as a way to signal from one commit that you are
leaving a bug that a later commit will fix:

```rust,ignore
if foo {
    return true; // TODO wrong, but will be fixed in a later commit
}
```

<a name="cio"></a>

# Using crates from crates.io

It is allowed to use crates from crates.io, though external
dependencies should not be added gratuitously. All such crates must
have a suitably permissive license. There is an automatic check which
inspects the Cargo metadata to ensure this.

<a name="er"></a>

# How to structure your PR

How you prepare the commits in your PR can make a big difference for the
reviewer.  Here are some tips.

**Isolate "pure refactorings" into their own commit.** For example, if
you rename a method, then put that rename into its own commit, along
with the renames of all the uses.

**More commits is usually better.** If you are doing a large change,
it's almost always better to break it up into smaller steps that can
be independently understood. The one thing to be aware of is that if
you introduce some code following one strategy, then change it
dramatically (versus adding to it) in a later commit, that
'back-and-forth' can be confusing.

**If you run rustfmt and the file was not already formatted, isolate
that into its own commit.** This is really the same as the previous
rule, but it's worth highlighting. It's ok to rustfmt files, but since
we do not currently run rustfmt all the time, that can introduce a lot
of noise into your commit. Please isolate that into its own
commit. This also makes rebases a lot less painful, since rustfmt
tends to cause a lot of merge conflicts, and having those isolated
into their own commit makes them easier to resolve.

**No merges.** We do not allow merge commits into our history, other
than those by bors. If you get a merge conflict, rebase instead via a
command like `git rebase -i rust-lang/master` (presuming you use the
name `rust-lang` for your remote).

**Individual commits do not have to build (but it's nice).** We do not
require that every intermediate commit successfully builds – we only
expect to be able to bisect at a PR level. However, if you *can* make
individual commits build, that is always helpful.

# Naming conventions

Apart from normal Rust style/naming conventions, there are also some specific
to the compiler.

- `cx` tends to be short for "context" and is often used as a suffix. For
  example, `tcx` is a common name for the [Typing Context][tcx].

- [`'tcx` and `'gcx`][tcx] are used as the lifetime names for the Typing
  Context.

- Because `crate` is a keyword, if you need a variable to represent something
  crate-related, often the spelling is changed to `krate`.

[tcx]: ./ty.md
