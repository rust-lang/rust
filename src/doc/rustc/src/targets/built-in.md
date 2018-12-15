# Built-in Targets

`rustc` ships with the ability to compile to many targets automatically, we
call these "built-in" targets, and they generally correspond to targets that
the team is supporting directly.

To see the list of built-in targets, you can run `rustc --print target-list`,
or look at [the API
docs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_target/spec/index.html#modules).
Each module there defines a builder for a particular target.