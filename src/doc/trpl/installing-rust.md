% Installing Rust

The first step to using Rust is to install it. Generally speaking, you’ll need
an internet connection to run the commands in this chapter, as we’ll be
downloading Rust from the internet.

We’ll be showing off a number of commands using a terminal, and those lines all
start with `$`. We don't need to type in the `$`s, they are there to indicate
the start of each command. We’ll see many tutorials and examples around the web
that follow this convention: `$` for commands run as our regular user, and `#`
for commands we should be running as an administrator.

## Platform support

The Rust compiler runs on, and compiles to, a great number of platforms, though
not all platforms are equally supported. Rust's support levels are organized
into three tiers, each with a different set of guarantees.

Platforms are identified by their "target triple" which is the string to inform
the compiler what kind of output should be produced. The columns below indicate
whether the corresponding component works on the specified platform.

### Tier 1

Tier 1 platforms can be thought of as "guaranteed to build and work".
Specifically they will each satisfy the following requirements:

* Automated testing is set up to run tests for the platform.
* Landing changes to the `rust-lang/rust` repository's master branch is gated on
  tests passing.
* Official release artifacts are provided for the platform.
* Documentation for how to use and how to build the platform is available.

|  Target                       | std |rustc|cargo| notes                      |
|-------------------------------|-----|-----|-----|----------------------------|
| `x86_64-pc-windows-msvc`      |  ✓  |  ✓  |  ✓  | 64-bit MSVC (Windows 7+)   |
| `i686-pc-windows-gnu`         |  ✓  |  ✓  |  ✓  | 32-bit MinGW (Windows 7+)  |
| `x86_64-pc-windows-gnu`       |  ✓  |  ✓  |  ✓  | 64-bit MinGW (Windows 7+)  |
| `i686-apple-darwin`           |  ✓  |  ✓  |  ✓  | 32-bit OSX (10.7+, Lion+)  |
| `x86_64-apple-darwin`         |  ✓  |  ✓  |  ✓  | 64-bit OSX (10.7+, Lion+)  |
| `i686-unknown-linux-gnu`      |  ✓  |  ✓  |  ✓  | 32-bit Linux (2.6.18+)     |
| `x86_64-unknown-linux-gnu`    |  ✓  |  ✓  |  ✓  | 64-bit Linux (2.6.18+)     |

### Tier 2

Tier 2 platforms can be thought of as "guaranteed to build". Automated tests
are not run so it's not guaranteed to produce a working build, but platforms
often work to quite a good degree and patches are always welcome! Specifically,
these platforms are required to have each of the following:

* Automated building is set up, but may not be running tests.
* Landing changes to the `rust-lang/rust` repository's master branch is gated on
  platforms **building**. Note that this means for some platforms only the
  standard library is compiled, but for others the full bootstrap is run.
* Official release artifacts are provided for the platform.

|  Target                       | std |rustc|cargo| notes                      |
|-------------------------------|-----|-----|-----|----------------------------|
| `i686-pc-windows-msvc`        |  ✓  |  ✓  |  ✓  | 32-bit MSVC (Windows 7+)   |

### Tier 3

Tier 3 platforms are those which Rust has support for, but landing changes is
not gated on the platform either building or passing tests. Working builds for
these platforms may be spotty as their reliability is often defined in terms of
community contributions. Additionally, release artifacts and installers are not
provided, but there may be community infrastructure producing these in
unofficial locations.

|  Target                       | std |rustc|cargo| notes                      |
|-------------------------------|-----|-----|-----|----------------------------|
| `x86_64-unknown-linux-musl`   |  ✓  |     |     | 64-bit Linux with MUSL     |
| `arm-linux-androideabi`       |  ✓  |     |     | ARM Android                |
| `i686-linux-android`          |  ✓  |     |     | 32-bit x86 Android         |
| `aarch64-linux-android`       |  ✓  |     |     | ARM64 Android              |
| `arm-unknown-linux-gnueabi`   |  ✓  |  ✓  |     | ARM Linux (2.6.18+)        |
| `arm-unknown-linux-gnueabihf` |  ✓  |  ✓  |     | ARM Linux (2.6.18+)        |
| `aarch64-unknown-linux-gnu`   |  ✓  |     |     | ARM64 Linux (2.6.18+)      |
| `mips-unknown-linux-gnu`      |  ✓  |     |     | MIPS Linux (2.6.18+)       |
| `mipsel-unknown-linux-gnu`    |  ✓  |     |     | MIPS (LE) Linux (2.6.18+)  |
| `powerpc-unknown-linux-gnu`   |  ✓  |     |     | PowerPC Linux (2.6.18+)    |
| `i386-apple-ios`              |  ✓  |     |     | 32-bit x86 iOS             |
| `x86_64-apple-ios`            |  ✓  |     |     | 64-bit x86 iOS             |
| `armv7-apple-ios`             |  ✓  |     |     | ARM iOS                    |
| `armv7s-apple-ios`            |  ✓  |     |     | ARM iOS                    |
| `aarch64-apple-ios`           |  ✓  |     |     | ARM64 iOS                  |
| `i686-unknown-freebsd`        |  ✓  |  ✓  |     | 32-bit FreeBSD             |
| `x86_64-unknown-freebsd`      |  ✓  |  ✓  |     | 64-bit FreeBSD             |
| `x86_64-unknown-openbsd`      |  ✓  |  ✓  |     | 64-bit OpenBSD             |
| `x86_64-unknown-netbsd`       |  ✓  |  ✓  |     | 64-bit NetBSD              |
| `x86_64-unknown-bitrig`       |  ✓  |  ✓  |     | 64-bit Bitrig              |
| `x86_64-unknown-dragonfly`    |  ✓  |  ✓  |     | 64-bit DragonFlyBSD        |
| `x86_64-rumprun-netbsd`       |  ✓  |     |     | 64-bit NetBSD Rump Kernel  |
| `i686-pc-windows-msvc` (XP)   |  ✓  |     |     | Windows XP support         |
| `x86_64-pc-windows-msvc` (XP) |  ✓  |     |     | Windows XP support         |

Note that this table can be expanded over time, this isn't the exhaustive set of
tier 3 platforms that will ever be!

## Installing on Linux or Mac

If we're on Linux or a Mac, all we need to do is open a terminal and type this:

```bash
$ curl -sf -L https://static.rust-lang.org/rustup.sh | sh
```

This will download a script, and stat the installation. If it all goes well,
you’ll see this appear:

```
Welcome to Rust.

This script will download the Rust compiler and its package manager, Cargo, and
install them to /usr/local. You may install elsewhere by running this script
with the --prefix=<path> option.

The installer will run under ‘sudo’ and may ask you for your password. If you do
not want the script to run ‘sudo’ then pass it the --disable-sudo flag.

You may uninstall later by running /usr/local/lib/rustlib/uninstall.sh,
or by running this script again with the --uninstall flag.

Continue? (y/N) 
```

From here, press `y` for ‘yes’, and then follow the rest of the prompts.

## Installing on Windows

If you're on Windows, please download the appropriate [installer][install-page].

[install-page]: https://www.rust-lang.org/install.html

## Uninstalling

If you ever need to uninstall Rust for any reason, refer to this section to
find out how. It's actually as easy as installing. On Linux or Mac, just run
the uninstall script:

```bash
$ sudo /usr/local/lib/rustlib/uninstall.sh
```

If we used the Windows installer, we can re-run the `.msi` and it will give us
an uninstall option.

## Troubleshooting

If we've got Rust installed, we can open up a shell, and type this:

```bash
$ rustc --version
```

You should see the version number, commit hash, and commit date.

If you do, Rust has been installed successfully! Congrats!

This installer also installs a copy of the documentation locally, so we can
read it offline. On UNIX systems, `/usr/local/share/doc/rust` is the location.
On Windows, it's in a `share/doc` directory, inside the directory to which Rust
was installed.

If you don't and you're on Windows, check that Rust is in your %PATH% system
variable. If it isn't, run the installer again, select "Change" on the "Change,
repair, or remove installation" page and ensure "Add to PATH" is installed on
the local hard drive.

If not, there are a number of places where we can get help. The easiest is
[the #rust IRC channel on irc.mozilla.org][irc], which we can access through
[Mibbit][mibbit]. Click that link, and we'll be chatting with other Rustaceans
(a silly nickname we call ourselves) who can help us out. Other great resources
include [the user’s forum][users], and [Stack Overflow][stackoverflow].

[irc]: irc://irc.mozilla.org/#rust
[mibbit]: http://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust
[users]: https://users.rust-lang.org/
[stackoverflow]: http://stackoverflow.com/questions/tagged/rust
