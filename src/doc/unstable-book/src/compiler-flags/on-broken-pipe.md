# `on-broken-pipe`

--------------------

The tracking issue for this feature is: [#97889]

Note: The ui for this feature was previously an attribute named `#[unix_sigpipe = "..."]`.

[#97889]: https://github.com/rust-lang/rust/issues/97889

---


## Overview

The `-Zon-broken-pipe=...` compiler flag can be used to specify how libstd shall setup `SIGPIPE` on Unix platforms before invoking `fn main()`. This flag is ignored on non-Unix targets. The flag can be used with three different values or be omitted entirely. It affects `SIGPIPE` before `fn main()` and before children get `exec()`'ed:

| Compiler flag              | `SIGPIPE` before `fn main()` | `SIGPIPE` before child `exec()` |
|----------------------------|------------------------------|---------------------------------|
| not used                   | `SIG_IGN`                    | `SIG_DFL`                       |
| `-Zon-broken-pipe=kill`    | `SIG_DFL`                    | not touched                     |
| `-Zon-broken-pipe=error`   | `SIG_IGN`                    | not touched                     |
| `-Zon-broken-pipe=inherit` | not touched                  | not touched                     |


## `-Zon-broken-pipe` not used

If `-Zon-broken-pipe` is not used, libstd will behave in the manner it has since 2014, before Rust 1.0. `SIGPIPE` will be set to `SIG_IGN` before `fn main()` and result in `EPIPE` errors which are converted to `std::io::ErrorKind::BrokenPipe`.

When spawning child processes, `SIGPIPE` will be set to `SIG_DFL` before doing the underlying `exec()` syscall.


## `-Zon-broken-pipe=kill`

Set the `SIGPIPE` handler to `SIG_DFL` before invoking `fn main()`. This will result in your program getting killed if it tries to write to a closed pipe. This is normally what you want if your program produces textual output.

When spawning child processes, `SIGPIPE` will not be touched. This normally means child processes inherit `SIG_DFL` for `SIGPIPE`.

### Example

```rust,no_run
fn main() {
    loop {
        println!("hello world");
    }
}
```

```console
$ rustc -Zon-broken-pipe=kill main.rs
$ ./main | head -n1
hello world
```

## `-Zon-broken-pipe=error`

Set the `SIGPIPE` handler to `SIG_IGN` before invoking `fn main()`. This will result in `ErrorKind::BrokenPipe` errors if you program tries to write to a closed pipe. This is normally what you want if you for example write socket servers, socket clients, or pipe peers.

When spawning child processes, `SIGPIPE` will not be touched. This normally means child processes inherit `SIG_IGN` for `SIGPIPE`.

### Example

```rust,no_run
fn main() {
    loop {
        println!("hello world");
    }
}
```

```console
$ rustc -Zon-broken-pipe=error main.rs
$ ./main | head -n1
hello world
thread 'main' panicked at library/std/src/io/stdio.rs:1118:9:
failed printing to stdout: Broken pipe (os error 32)
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

## `-Zon-broken-pipe=inherit`

Leave `SIGPIPE` untouched before entering `fn main()`. Unless the parent process has changed the default `SIGPIPE` handler from `SIG_DFL` to something else, this will behave the same as `-Zon-broken-pipe=kill`.

When spawning child processes, `SIGPIPE` will not be touched. This normally means child processes inherit `SIG_DFL` for `SIGPIPE`.
