This crate is regularly synced with its mirror in the rustc repo at `compiler/rustc_smir`.

We use `git subtree` for this to preserve commits and allow the rustc repo to
edit these crates without having to touch this repo. This keeps the crates compiling
while allowing us to independently work on them here. The effort of keeping them in
sync is pushed entirely onto us, without affecting rustc workflows negatively.
This may change in the future, but changes to policy should only be done via a
compiler team MCP.

## Instructions for syncing

### Updating this repository

In the rustc repo, execute

```
git subtree push --prefix=compiler/rustc_smir url_to_your_fork_of_project_stable_mir some_feature_branch
```

and then open a PR of your `some_feature_branch` against https://github.com/rust-lang/project-stable-mir

### Updating the rustc librar


In the rustc repo, execute

```
git subtree pull --prefix=compiler/rustc_smir https://github.com/rust-lang/project-stable-mir smir
```

Note: only ever sync to rustc from the project-stable-mir's `smir` branch. Do not sync with your own forks.

Then open a PR against rustc just like a regular PR.
