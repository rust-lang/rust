# Publishing to crates.io

Publishing `compiler-builtins` to crates.io takes a few steps unfortunately.
It's not great, but it works for now. PRs to improve this process would be
greatly appreciated!

1. Make sure you've got a clean working tree and it's updated with the latest
   changes on `master`
2. Edit `Cargo.toml` to bump the version number
3. Commit this change
4. Run `git tag` to create a tag for this version
5. Delete the `libm/Cargo.toml` file
6. Run `cargo +nightly publish`
7. Push the tag
8. Push the commit
9. Undo changes to `Cargo.toml` and the `libm` submodule
