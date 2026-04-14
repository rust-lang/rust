# smallsh upstream

- origin: `https://github.com/loganintech/smallsh`
- pinned commit: `8715f2d29ebaf05bdd3d136e36417da6141c15c3`
- refresh command: `./scripts/sync_smallsh.sh`

## Layout

- [`upstream/`](/home/dancxjo/src/thing-os/userspace/smallsh/upstream) is an exact snapshot of the pinned upstream commit.
- [`src/`](/home/dancxjo/src/thing-os/userspace/smallsh/src) is the Thing-OS port that keeps the upstream module split but swaps in Thing-OS runtime assumptions.

## Port notes

- The upstream crate used `dirs`, `std::env::set_current_dir`, and a background reaper thread.
- The Thing-OS port removes the external dependency, keeps cwd as shell state, prefixes bare commands with `/` like the existing [`sh`](/home/dancxjo/src/thing-os/userspace/sh/src/main.rs), and reaps background jobs before each prompt instead of using a helper thread.
- The crate is built as a Thing-OS userspace binary with `stem` runtime glue and is included in the default ISO/HDD program list.
