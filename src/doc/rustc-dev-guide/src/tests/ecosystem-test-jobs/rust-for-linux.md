# Rust for Linux integration tests

[Rust for Linux](https://rust-for-linux.com/) (RfL) is an effort for adding
support for the Rust programming language into the Linux kernel.

## What to do if the Rust for Linux job breaks?

If a PR breaks the Rust for Linux CI job, then:

- If the breakage was unintentional and seems spurious, then let [RfL][rfl-ping]
  know and retry.
    - If the PR is urgent and retrying doesn't fix it, then disable the CI job
      temporarily (comment out the `image: x86_64-rust-for-linux` job in
      `src/ci/github-actions/jobs.yml`).
- If the breakage was unintentional, then change the PR to resolve the breakage.
- If the breakage was intentional, then let [RfL][rfl-ping] know and discuss
  what will the kernel need to change.
    - If the PR is urgent, then disable the CI job temporarily (comment out
      the `image: x86_64-rust-for-linux` job in `src/ci/github-actions/jobs.yml`).
    - If the PR can wait a few days, then wait for RfL maintainers to provide a
      new Linux kernel commit hash with the needed changes done, and apply it to
      the PR, which would confirm the changes work (update the  `LINUX_VERSION`
      environment variable in `src/ci/docker/scripts/rfl-build.sh`).

If you need to contact the RfL developers, you can ping the [Rust for Linux][rfl-ping]
ping group to ask for help:

```text
@rustbot ping rfl
```

## Building Rust for Linux in CI

Rust for Linux builds as part of the suite of bors tests that run before a pull
request is merged.

The workflow builds a stage1 sysroot of the Rust compiler, downloads the Linux
kernel, and tries to compile several Rust for Linux drivers and examples using
this sysroot. RfL uses several unstable compiler/language features, therefore
this workflow notifies us if a given compiler change would break it.

If you are worried that a pull request might break the Rust for Linux builder
and want to test it out before submitting it to the bors queue, simply ask
bors to run the try job that builds the Rust for Linux integration:
`@bors try jobs=x86_64-rust-for-linux`.

[rfl-ping]: ../../notification-groups/rust-for-linux.md
