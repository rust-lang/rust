% Nightly Rust

Rust provides three distribution channels for Rust: nightly, beta, and stable.
Unstable features are only available on nightly Rust. For more details on this
process, see [this post](http://blog.rust-lang.org/2014/10/30/Stability.html).

To install nightly Rust, you can use `rustup.sh`:

```bash
$ curl -s https://static.rust-lang.org/rustup.sh | sudo sh -s -- --channel=nightly
```

If you're concerned about the [potential insecurity](http://curlpipesh.tumblr.com/) of using `curl | sudo sh`,
please keep reading and see our disclaimer below. And feel free to use a two-step version of the installation and examine our installation script:

```bash
$ curl -f -L https://static.rust-lang.org/rustup.sh -O
$ sudo sh rustup.sh --channel=nightly
```

If you're on Windows, please download either the [32-bit
installer](https://static.rust-lang.org/dist/rust-nightly-i686-pc-windows-gnu.exe)
or the [64-bit
installer](https://static.rust-lang.org/dist/rust-nightly-x86_64-pc-windows-gnu.exe)
and run it.

If you decide you don't want Rust anymore, we'll be a bit sad, but that's okay.
Not every programming language is great for everyone. Just run the uninstall
script:

```bash
$ sudo /usr/local/lib/rustlib/uninstall.sh
```

If you used the Windows installer, just re-run the `.exe` and it will give you
an uninstall option.

You can re-run this script any time you want to update Rust. Which, at this
point, is often. Rust is still pre-1.0, and so people assume that you're using
a very recent Rust.

This brings me to one other point: some people, and somewhat rightfully so, get
very upset when we tell you to `curl | sudo sh`. And they should be! Basically,
when you do this, you are trusting that the good people who maintain Rust
aren't going to hack your computer and do bad things. That's a good instinct!
If you're one of those people, please check out the documentation on [building
Rust from Source](https://github.com/rust-lang/rust#building-from-source), or
[the official binary downloads](http://www.rust-lang.org/install.html). And we
promise that this method will not be the way to install Rust forever: it's just
the easiest way to keep people updated while Rust is in its alpha state.

