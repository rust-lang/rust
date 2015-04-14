% Installing Rust

The first step to using Rust is to install it! There are a number of ways to
install Rust, but first we ought to go over the officially supported arch's.

* Windows (7, 8, Server 2008 R2)***
* Linux (2.6.18 or later, various distributions), x86 and x86-64
* OSX 10.7 (Lion) or greater, x86 and x86-64

We extensively test Rust on these platforms, there are plans to support Android
in the future, that documentation will be public as it becomes available.

***Rust considers Windows to be a first-class platform upon release, but if we're 
truly honest, the Windows experience isn't as integrated as the Linux/OS X experience is.
Each and every commit is tested against Windows just like any other platform, if anything
does not work it is a bug.  Please let us know if you encounter one we'll squash it for you.

Rust also installs a copy of the documentation locally, so you can
read it offline. On UNIX systems, `/usr/local/share/doc/rust` is the location.
On Windows, it's in a `share/doc` directory, inside wherever you installed Rust
to.(Default C:\Program Files\Rust beta 1.0\ )

If you're on Windows, please download either the [32-bit installer][win32] or
the [64-bit installer][win64] and run it.

[win32]: https://static.rust-lang.org/dist/rust-1.0.0-beta-i686-pc-windows-gnu.msi
[win64]: https://static.rust-lang.org/dist/rust-1.0.0-beta-x86_64-pc-windows-gnu.msi

If you would like to build from source, please check out the
documentation on [building Rust from Source][from source], there is always [the official
binary downloads][install page]as well. 

[from source]: https://github.com/rust-lang/rust#building-from-source
[install page]: http://www.rust-lang.org/install.html

If you're on Linux or Mac rust has a tool called ['multirust'](https://github.com/brson/multirust/README.md) for managing
Rust, Cargo, and multiple custom toolchains; mappable to any working directory. Multirust is still a work in progress
and is not a part of Rust:master.(Yet)

Rust is signed with a [PGPKey](../trpl/pgp.key)!

To install 'multirust' from Github first move to your project directory.

````bash
$ cd /home/$usr/rust/
```
Next we'll want to get the sourcecode from Github.

```bash
$ git clone --recursive https://github.com/brson/multirust
```

We will then move to the new multirust directory.

```bash
$ cd ./rust/multirust
```

Now we will initialize the submodule

```bash
$ git submodule update --init
```

And finally we will install 'multirust' which will handle the installation of Rust and Cargo!
(This step will require your root password)

```bash
./build.sh && sudo ./install.sh
```
See [multirust](../doc/trpl/multirust_advsetup.md) advanced setup for toolchain configuration options and everything you ever wanted to know and more about multirust.

Otherwise if you are working on your own machine and trust the development team.

Install multirust via

```bash
$ sudo curl -sf https://raw.githubusercontent.com/brson/multirust/master/blastoff.sh | sh
```
Or Rust proper (minus multirust)

```bash
$ curl -sf -L https://static.rust-lang.org/rustup.sh | sudo sh
```

## Uninstalling

If you decide you don't want Rust anymore, we'll make sure to keep your seat warm until you get back.

If you used the Windows installer, just re-run the `.msi` and it will give you an uninstall option.

If you maintain your Rust installation via multirust uninstall is simple.

```bash
$ sudo ./install.sh --uninstall
```
Note: If you added additional install options as described on the [multirust](../doc/trpl/multirust_advsetup.md)
page you will want to see that same page for the additional uninstallation options.

If you did not use multirust the command is 

```bash
$ sudo /usr/local/lib/rustlib/uninstall.sh
```

##Installation test and Upgrade!

Upgrading to the latest nightly is easy!

For multirust:

```bash
$ multirust upgrade nightly
```
Note: We recommend the beta unless you absolutely need features in the nightly.

For source/direct installation upgrades 

```bash
#Hopefully deprecated by launch
$ curl -sf -L https://static.rust-lang.org/rustup.sh | sudo sh
```
If you didn't install multirust and want to get the current nightly.
(Please just get multirust instead of doing this, it will make it way simpler to debug when you can cross compile with different versions of nightlies.)

```bash
$ curl -sf -L https://static.rustlang.org/rustup.sh | sudo sh -s -- --channel=nightly
```

To test if installation was successful use:

```bash
$ rustc --version
```

You should see the version number, commit hash, commit date and build date:

```bash
rustc 1.0.0-beta (9854143cb 2015-04-02) (built 2015-04-02)
```

If you did, Rust has been installed successfully! Yay!

If not, there are a number of places where you can get help. The easiest is
[the #rust IRC channel on irc.mozilla.org][irc], which you can access through
[Mibbit][mibbit]. Click that link, and you'll be chatting with other Rustaceans
(a silly nickname we call ourselves), and we can help you out. Other great
resources include [the user’s forum][users], [Stack Overflow][stackoverflow],
and [Reddit][reddit].  Project specific IRC channels can be found in the reddit sidebar.

[reddit]: http://www.reddit.com/r/rust
[irc]: irc://irc.mozilla.org/#rust
[mibbit]: http://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust
[users]: http://users.rust-lang.org/ 
[stack overflow]: http://stackoverflow.com/questions/tagged/rust
