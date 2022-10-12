# Backport Changes

Sometimes it is necessary to backport changes to the beta release of Clippy.
Backports in Clippy are rare and should be approved by the Clippy team. For
example, a backport is done, if a crucial ICE was fixed or a lint is broken to a
point, that it has to be disabled, before landing on stable.

Backports are done to the `beta` branch of Clippy. Backports to stable Clippy
releases basically don't exist, since this would require a Rust point release,
which is almost never justifiable for a Clippy fix.


## Backport the changes

Backports are done on the beta branch of the Clippy repository.

```bash
# Assuming the current directory corresponds to the Clippy repository
$ git checkout beta
$ git checkout -b backport
$ git cherry-pick <SHA>  # `<SHA>` is the commit hash of the commit(s), that should be backported
$ git push origin backport
```

Now you should test that the backport passes all the tests in the Rust
repository. You can do this with:

```bash
# Assuming the current directory corresponds to the Rust repository
$ git checkout beta
$ git subtree pull -p src/tools/clippy https://github.com/<your-github-name>/rust-clippy backport
$ ./x.py test src/tools/clippy
```

Should the test fail, you can fix Clippy directly in the Rust repository. This
has to be first applied to the Clippy beta branch and then again synced to the
Rust repository, though. The easiest way to do this is:

```bash
# In the Rust repository
$ git diff --patch --relative=src/tools/clippy > clippy.patch
# In the Clippy repository
$ git apply /path/to/clippy.patch
$ git add -u
$ git commit -m "Fix rustup fallout"
$ git push origin backport
```

After this, you can open a PR to the `beta` branch of the Clippy repository.


## Update Clippy in the Rust Repository

This step must be done, **after** the PR of the previous step was merged.

After the backport landed in the Clippy repository, the branch has to be synced
back to the beta branch of the Rust repository.

```bash
# Assuming the current directory corresponds to the Rust repository
$ git checkout beta
$ git checkout -b clippy_backport
$ git subtree pull -p src/tools/clippy https://github.com/rust-lang/rust-clippy beta
$ git push origin clippy_backport
```

Make sure to test the backport in the Rust repository before opening a PR. This
is done with `./x.py test src/tools/clippy`. If that passes all tests, open a PR
to the `beta` branch of the Rust repository. In this PR you should tag the
Clippy team member, that agreed to the backport or the `@rust-lang/clippy` team.
Make sure to add `[beta]` to the title of the PR.
