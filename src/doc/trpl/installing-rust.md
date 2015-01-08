% Installing Rust

The first step to using Rust is to install it! There are a number of ways to
install Rust, but the easiest is to use the `rustup` script. If you're on
Linux or a Mac, all you need to do is this (note that you don't need to type
in the `$`s, they just indicate the start of each command):

```bash
$ curl -L https://static.rust-lang.org/rustup.sh | sudo sh
```

If you're concerned about the [potential insecurity](http://curlpipesh.tumblr.com/) of using `curl | sudo sh`,
please keep reading and see our disclaimer below. And feel free to use a two-step version of the installation and examine our installation script:

```bash
$ curl -L https://static.rust-lang.org/rustup.sh -O
$ sudo sh rustup.sh
```

If you're on Windows, please download either the [32-bit
installer](https://static.rust-lang.org/dist/rust-nightly-i686-pc-windows-gnu.exe)
or the [64-bit
installer](https://static.rust-lang.org/dist/rust-nightly-x86_64-pc-windows-gnu.exe)
and run it.

If you decide you don't want Rust anymore, we'll be a bit sad, but that's okay.
Not every programming language is great for everyone. Just pass an argument to
the script:

```bash
$ curl -s https://static.rust-lang.org/rustup.sh | sudo sh -s -- --uninstall
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

Oh, we should also mention the officially supported platforms:

* Windows (7, 8, Server 2008 R2)
* Linux (2.6.18 or later, various distributions), x86 and x86-64
* OSX 10.7 (Lion) or greater, x86 and x86-64

We extensively test Rust on these platforms, and a few others, too, like
Android. But these are the ones most likely to work, as they have the most
testing.

Finally, a comment about Windows. Rust considers Windows to be a first-class
platform upon release, but if we're honest, the Windows experience isn't as
integrated as the Linux/OS X experience is. We're working on it! If anything
does not work, it is a bug. Please let us know if that happens. Each and every
commit is tested against Windows just like any other platform.

If you've got Rust installed, you can open up a shell, and type this:

```bash
$ rustc --version
```

You should see some output that looks something like this:

```bash
rustc 1.0.0-nightly (f11f3e7ba 2015-01-04 20:02:14 +0000)
```

If you did, Rust has been installed successfully! Congrats!

If not, there are a number of places where you can get help. The easiest is
[the #rust IRC channel on irc.mozilla.org](irc://irc.mozilla.org/#rust), which
you can access through
[Mibbit](http://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust). Click
that link, and you'll be chatting with other Rustaceans (a silly nickname we
call ourselves), and we can help you out. Other great resources include [our
forum](http://discuss.rust-lang.org/), [the /r/rust
subreddit](http://www.reddit.com/r/rust), and [Stack
Overflow](http://stackoverflow.com/questions/tagged/rust).
