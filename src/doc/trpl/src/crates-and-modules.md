% Crates and Modules

Rust features a strong module system, but it works a bit differently than in
other programming languages. Rust's module system has two main components:
**crate**s and **module**s.

A crate is Rust's unit of independent compilation. Rust always compiles one
crate at a time, producing either a library or an executable. However, executables
usually depend on libraries, and many libraries depend on other libraries as well.
To support this, crates can depend on other crates.

Each crate contains a hierarchy of modules. This tree starts off with a single
module, called the **crate root**. Within the crate root, we can declare other
modules, which can contain other modules, as deeply as you'd like.

Note that we haven't mentioned anything about files yet. Rust does not impose a
particular relationship between your filesystem structure and your module
structure. That said, there is a conventional approach to how Rust looks for
modules on the file system, but it's also overridable.

Enough talk, let's build something! Let's make a new project called `modules`.

```{bash,ignore}
$ cd ~/projects
$ cargo new modules --bin
$ cd modules
```

Let's double check our work by compiling:

```{bash,notrust}
$ cargo run
   Compiling modules v0.0.1 (file:///home/you/projects/modules)
     Running `target/modules`
Hello, world!
```

Excellent! So, we already have a single crate here: our `src/main.rs` is a crate.
Everything in that file is in the crate root. A crate that generates an executable
defines a `main` function inside its root, as we've done here.

Let's define a new module inside our crate. Edit `src/main.rs` to look
like this:

```
fn main() {
    println!("Hello, world!")
}

mod hello {
    fn print_hello() {
        println!("Hello, world!")
    }
}
```

We now have a module named `hello` inside of our crate root. Modules use
`snake_case` naming, like functions and variable bindings.

Inside the `hello` module, we've defined a `print_hello` function. This will
also print out our hello world message. Modules allow you to split up your
program into nice neat boxes of functionality, grouping common things together,
and keeping different things apart. It's kinda like having a set of shelves:
a place for everything and everything in its place.

To call our `print_hello` function, we use the double colon (`::`):

```{rust,ignore}
hello::print_hello();
```

You've seen this before, with `io::stdin()` and `rand::random()`. Now you know
how to make your own. However, crates and modules have rules about
**visibility**, which controls who exactly may use the functions defined in a
given module. By default, everything in a module is private, which means that
it can only be used by other functions in the same module. This will not
compile:

```{rust,ignore}
fn main() {
    hello::print_hello();
}

mod hello {
    fn print_hello() {
        println!("Hello, world!")
    }
}
```

It gives an error:

```{notrust,ignore}
   Compiling modules v0.0.1 (file:///home/you/projects/modules)
src/main.rs:2:5: 2:23 error: function `print_hello` is private
src/main.rs:2     hello::print_hello();
                  ^~~~~~~~~~~~~~~~~~
```

To make it public, we use the `pub` keyword:

```{rust}
fn main() {
    hello::print_hello();
}

mod hello {
    pub fn print_hello() {
        println!("Hello, world!")
    }
}
```

Usage of the `pub` keyword is sometimes called 'exporting', because
we're making the function available for other modules. This will work:

```{notrust,ignore}
$ cargo run
   Compiling modules v0.0.1 (file:///home/you/projects/modules)
     Running `target/modules`
Hello, world!
```

Nice! There are more things we can do with modules, including moving them into
their own files. This is enough detail for now.
