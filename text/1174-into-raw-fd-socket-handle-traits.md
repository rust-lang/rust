- Feature Name: into-raw-fd-socket-handle-traits
- Start Date: 2015-06-24
- RFC PR: [rust-lang/rfcs#1174](https://github.com/rust-lang/rfcs/pull/1174)
- Rust Issue: [rust-lang/rust#27062](https://github.com/rust-lang/rust/issues/27062)

# Summary

Introduce and implement `IntoRaw{Fd, Socket, Handle}` traits to complement the
existing `AsRaw{Fd, Socket, Handle}` traits already in the standard library.

# Motivation

The `FromRaw{Fd, Socket, Handle}` traits each take ownership of the provided
handle, however, the `AsRaw{Fd, Socket, Handle}` traits do not give up
ownership. Thus, converting from one handle wrapper to another (for example
converting an open `fs::File` to a `process::Stdio`) requires the caller to
either manually `dup` the handle, or `mem::forget` the wrapper, which
is unergonomic and can be prone to mistakes.

Traits such as `IntoRaw{Fd, Socket, Handle}` will allow for easily transferring
ownership of OS handles, and it will allow wrappers to perform any
cleanup/setup as they find necessary.

# Detailed design

The `IntoRaw{Fd, Socket, Handle}` traits will behave exactly like their
`AsRaw{Fd, Socket, Handle}` counterparts, except they will consume the wrapper
before transferring ownership of the handle.

Note that these traits should **not** have a blanket implementation over `T:
AsRaw{Fd, Socket, Handle}`: these traits should be opt-in so that implementors
can decide if leaking through `mem::forget` is acceptable or another course of
action is required.

```rust
// Unix
pub trait IntoRawFd {
    fn into_raw_fd(self) -> RawFd;
}

// Windows
pub trait IntoRawSocket {
    fn into_raw_socket(self) -> RawSocket;
}

// Windows
pub trait IntoRawHandle {
    fn into_raw_handle(self) -> RawHandle;
}
```

# Drawbacks

This adds three new traits and methods which would have to be maintained.

# Alternatives

Instead of defining three new traits we could instead use the
`std::convert::Into<T>` trait over the different OS handles. However, this
approach will not offer a duality between methods such as
`as_raw_fd()`/`into_raw_fd()`, but will instead be `as_raw_fd()`/`into()`.

Another possibility is defining both the newly proposed traits as well as the
`Into<T>` trait over the OS handles letting the caller choose what they prefer.

# Unresolved questions

None at the moment.
