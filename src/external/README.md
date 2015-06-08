## External crates

All crates in this directory are hosted externally from this repository and are
imported via the standard `git-subtree` command. These crates **should not** be
edited directly, but instead changes to should go upstream and then be pulled
into these crates.

Crates here are listed in the `EXTERNAL_CRATES` array in `mk/crates.mk` and are
built via the standard build system.

### Adding a new external crate

1. Make sure the crate has the appropriate `#![cfg_attr]` annotations to make
   the crate unstable in the distribution with a message pointing to crates.io.
   See the existing crates in the `src/external` folder for examples.

2. To add a new crate `foo` to this folder, first execute the following:

   ```sh
   git subtree add -P src/external/foo https://github.com/bar/foo master --squash
   ```

  This will check out the crate into this folder, squashing the entire history
  into one commit (the rust-lang/rust repo doesn't need the whole history of the
  crate).

3. Next, edit `mk/crates.mk` appropriately by modifying `EXTERNAL_CRATES` and
   possibly some other crates and/or dependency lists.

4. Add the crate to `src/test/compile-fail-fulldeps/unstable-crates.rs` to
   ensure that it is unstable in the distribution.

### Updating an external crate

To pull in upstream changes to a library `foo`, execute the following

```sh
git subtree pull -P src/external/foo https://github.com/bar/foo master --squash
```

Similar to the addition process the `--squash` argument is provided to squash
all changes into one commit.
