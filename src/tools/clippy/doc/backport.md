# Backport Changes

Sometimes it is necessary to backport changes to the beta release of Clippy.
Backports in Clippy are rare and should be approved by the Clippy team. For
example, a backport is done, if a crucial ICE was fixed or a lint is broken to a
point, that it has to be disabled, before landing on stable.

Backports are done to the `beta` release of Clippy. Backports to stable Clippy
releases basically don't exist, since this would require a Rust point release,
which is almost never justifiable for a Clippy fix.


## Backport the changes

Backports are done on the beta branch of the Clippy repository.

```bash
# Assuming the current directory corresponds to the Clippy repository
$ git checkout beta
$ git checkout -b backport
$ git cherry-pick <SHA>  # `<SHA>` is the commit hash of the commit, that should be backported
$ git push origin backport
```

After this, you can open a PR to the `beta` branch of the Clippy repository.


## Update Clippy in the Rust Repository

This step must be done, **after** the PR of the previous step was merged.

After the backport landed in the Clippy repository, also the Clippy version on
the Rust `beta` branch has to be updated.

```bash
# Assuming the current directory corresponds to the Rust repository
$ git checkout beta
$ git checkout -b clippy_backport
$ pushd src/tools/clippy
$ git fetch
$ git checkout beta
$ popd
$ git add src/tools/clippy
§ git commit -m "Update Clippy"
$ git push origin clippy_backport
```

After this you can open a PR to the `beta` branch of the Rust repository. In
this PR you should tag the Clippy team member, that agreed to the backport or
the `@rust-lang/clippy` team. Make sure to add `[beta]` to the title of the PR.
