# Rust for Linux integration tests

[Rust for Linux](https://rust-for-linux.com/) (RfL) is an effort for adding support for the Rust programming
language into the Linux kernel.

## Building Rust for Linux in CI

Rust for Linux builds as part of the suite of bors tests that run before a pull request
is merged.

The workflow builds a stage1 sysroot of the Rust compiler, downloads the Linux kernel, and tries to compile several Rust for Linux drivers and examples using this sysroot. RfL uses several unstable compiler/language features, therefore this workflow notifies us if a given compiler change would break it.

If you are worried that a pull request might break the Rust for Linux builder and want
to test it out before submitting it to the bors queue, simply add this line to
your PR description:

> try-job: x86_64-rust-for-linux

Then when you `@bors try` it will pick the job that builds the Rust for Linux integration.

## What to do in case of failure

Currently, we use the following unofficial policy for handling failures caused by a change breaking the RfL integration:

- If the breakage was unintentional, then fix the PR.
- If the breakage was intentional, then let [RFL][rfl-ping] know and discuss what will the kernel need to change.
    - If the PR is urgent, then disable the test temporarily.
    - If the PR can wait a few days, then wait for RFL maintainers to provide a new Linux kernel commit hash with the needed changes done, and apply it to the PR, which would confirm the changes work.

If something goes wrong with the workflow, you can ping the [Rust for Linux][rfl-ping] ping group to ask for help.

[rfl-ping]: ../notification-groups/rust-for-linux.md
