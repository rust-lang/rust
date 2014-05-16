---
title: "Getting Started"
---

# Getting Started

Let's go through the process of installing Rust on your system and being able to compile your first program.


## Installation

Head over to [Rust's homepage](http://www.rust-lang.org/) and download the nightly version of Rust for your operating system. This should be a simple, one-click install.

## Using It

Now that Rust is installed, you'll have access to the `rustc` executable. That's the Rust compiler.

Let's create a new file called `hello_world.rs`. The Rust file extension, if you didn't know already, is `rs`.

``` {.rust}
// hello_world.rs
fn main() {
  println!("Hello World!");
}
```

Now you can compile it with:

```bash
$ rustc hello_world.rs -o helloworld
```

Let's run the hello world program written in Rust:

```
$ ./helloworld
Hello World!
```