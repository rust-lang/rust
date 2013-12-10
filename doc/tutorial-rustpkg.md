% Rust Packaging Tutorial

# Introduction

Sharing is caring. Rust comes with a tool, `rustpkg`, which allows you to
package up your Rust code and share it with other people. This tutorial will
get you started on all of the concepts and commands you need to give the gift
of Rust code to someone else.

# Installing External Packages

First, let's try to use an external package somehow. I've made a sample package
called `hello` to demonstrate how to do so.  Here's how `hello` is used:

~~~~
extern mod hello;

fn main() {
    hello::world();
}
~~~~

Easy! But if you try to compile this, you'll get an error:

~~~~ {.notrust}
$ rustc main.rs 
main.rs:1:0: 1:17 error: can't find crate for `hello`
main.rs:1 extern mod hello;
          ^~~~~~~~~~~~~~~~~
~~~~

This makes sense, as we haven't gotten it from anywhere yet!  Luckily for us,
`rustpkg` has an easy way to fetch others' code: the `install` command. It's
used like this:

~~~ {.notrust}
$ rustpkg install PKG_ID
~~~

This will install a package named `PKG_ID` into your current Rust environment.
I called it `PKG_ID` in this example because `rustpkg` calls this a 'package
identifier.' When using it with an external package like this, it's often a
URI fragment.  You see, Rust has no central authority for packages. You can
build your own `hello` library if you want, and that's fine. We'd both host
them in different places and different projects would rely on whichever version
they preferred.

To install the `hello` library, simply run this in your terminal:

~~~ {.notrust}
$ rustpkg install github.com/steveklabnik/hello
~~~

You should see a message that looks like this:

~~~ {.notrust}
note: Installed package github.com/steveklabnik/hello-0.1 to /some/path/.rust
~~~

Now, compiling our example should work:

~~~ {.notrust}
$ rustc main.rs
$ ./main 
Hello, world.
~~~

Simple! That's all it takes.

# Workspaces

Before we can talk about how to make packages of your own, you have to
understand the big concept with `rustpkg`: workspaces. A 'workspace' is simply
a directory that has certain sub-directories that `rustpkg` expects. Different
Rust projects will go into different workspaces.

A workspace consists of any directory that has the following
directories:

* `src`: The directory where all the source code goes.
* `build`: This directory contains all of the build output.
* `lib`: The directory where any libraries distributed with the package go.
* `bin`: This directory holds any binaries distributed with the package.

There are also default file names you'll want to follow as well:

* `main.rs`: A file that's going to become an executable.
* `lib.rs`: A file that's going to become a library.

# Building your own Package

Now that you've got workspaces down, let's build your own copy of `hello`. Go
to wherever you keep your personal projects, and let's make all of the
directories we'll need. I'll refer to this personal project directory as
`~/src` for the rest of this tutorial.

## Creating our workspace

~~~ {.notrust}
$ cd ~/src
$ mkdir -p hello/src/hello
$ cd hello
~~~

Easy enough! Let's do one or two more things that are nice to do:

~~~ {.notrust}
$ git init .
$ cat > README.md
# hello

A simple package for Rust.

## Installation

```
$ rustpkg install github.com/YOUR_USERNAME/hello
```
^D
$ cat > .gitignore
.rust
build
^D
$ git commit -am "Initial commit."
~~~

If you're not familliar with the `cat >` idiom, it will make files with the
text you type inside. Control-D (`^D`) ends the text for the file.

Anyway, we've got a README and a `.gitignore`. Let's talk about that
`.gitignore` for a minute: we are ignoring two directories, `build` and
`.rust`. `build`, as we discussed earlier, is for build artifacts, and we don't
want to check those into a repository. `.rust` is a directory that `rustpkg`
uses to keep track of its own settings, as well as the source code of any other
external packages that this workspace uses. This is where that `rustpkg
install` puts all of its files. Those are also not to go into our repository,
so we ignore it all as well.

Next, let's add a source file:

~~~
#[desc = "A hello world Rust package."];
#[license = "MIT"];

pub fn world() {
    println("Hello, world.");
}
~~~

Put this into `src/hello/lib.rs`. Let's talk about each of these attributes:

## Crate attributes for packages

`license` is equally simple: the license we want this code to have. I chose MIT
here, but you should pick whatever license makes the most sense for you.

`desc` is a description of the package and what it does. This should just be a
sentence or two.

## Building your package

Building your package is simple:

~~~ {.notrust}
$ rustpkg build hello
~~~

This will compile `src/hello/lib.rs` into a library. After this process
completes, you'll want to check out `build`:

~~~ {.notrust}
$ ls build/x86_64-unknown-linux-gnu/hello/
libhello-ed8619dad9ce7d58-0.1.0.so
~~~

This directory naming structure is called a 'build triple,' and is because I'm
on 64 bit Linux. Yours may differ based on platform.

You'll also notice that `src/hello/lib.rs` turned into
`libhello-ed8619dad9ce7d58-0.1.0.so`. This is a simple combination of the
library name, a hash of its content, and the version.

Now that your library builds, you'll want to commit:

~~~ {.notrust}
$ git add src
$ git commit -m "Adding source code."
~~~

If you're using GitHub, after creating the project, do this:

~~~ {.notrust}
$ git remote add origin git@github.com:YOUR_USERNAME/hello.git
$ git push origin -u master
~~~

Now you can install and use it! Go anywhere else in your filesystem:

~~~ {.notrust}
$ cd ~/src/foo
$ rustpkg install github.com/YOUR_USERNAME/hello
WARNING: The Rust package manager is experimental and may be unstable
note: Installed package github.com/YOUR_USERNAME/hello-0.1 to /home/yourusername/src/hello/.rust
~~~

That's it!

# Testing your Package

Testing your package is simple as well. First, let's change `src/hello/lib.rs` to contain
a function that can be sensibly tested:

~~~
#[desc = "A Rust package for determining whether unsigned integers are even."];
#[license = "MIT"];

pub fn is_even(i: uint) -> bool {
    i % 2 == 0
}
~~~

Once you've edited `lib.rs`, you can create a second crate file, `src/hello/test.rs`,
to put tests in:

~~~
#[license = "MIT"];
extern mod hello;
use hello::is_even;

#[test]
fn test_is_even() {
    assert!(is_even(0));
    assert!(!is_even(1));
    assert!(is_even(2));
}
~~~

Note that you have to import the crate you just created in `lib.rs` with the
`extern mod hello` directive. That's because you're putting the tests in a different
crate from the main library that you created.

Now, you can use the `rustpkg test` command to build this test crate (and anything else
it depends on) and run the tests, all in one step:

~~~ {.notrust}
$ rustpkg test hello
WARNING: The Rust package manager is experimental and may be unstable
note: Installed package hello-0.1 to /Users/tjc/.rust

running 1 test
test test_is_even ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
~~~

# More resources

There's a lot more going on with `rustpkg`, this is just to get you started.
Check out [the rustpkg manual](rustpkg.html) for the full details on how to
customize `rustpkg`.

A tag was created on GitHub specifically for `rustpkg`-related issues. You can
[see all the Issues for rustpkg
here](https://github.com/mozilla/rust/issues?direction=desc&labels=A-pkg&sort=created&state=open),
with bugs as well as new feature plans. `rustpkg` is still under development,
and so may be a bit flaky at the moment.

You may also want to check out [this blog
post](http://tim.dreamwidth.org/1820526.html), which contains some of the early
design decisions and justifications.
