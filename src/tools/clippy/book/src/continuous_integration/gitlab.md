# GitLab CI

You can add Clippy to GitLab CI by using the latest stable [rust docker image](https://hub.docker.com/_/rust),
as it is shown in the `.gitlab-ci.yml` CI configuration file below,

```yml
# Make sure CI fails on all warnings, including Clippy lints
variables:
  RUSTFLAGS: "-Dwarnings"

clippy_check:
  image: rust:latest
  script:
    - rustup component add clippy
    - cargo clippy --all-targets --all-features
```
