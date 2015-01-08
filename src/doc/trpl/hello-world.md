% Hello, world!

Now that you have Rust installed, let's write your first Rust program. It's
traditional to make your first program in any new language one that prints the
text "Hello, world!" to the screen. The nice thing about starting with such a
simple program is that you can verify that your compiler isn't just installed,
but also working properly. And printing information to the screen is a pretty
common thing to do.

The first thing that we need to do is make a file to put our code in. I like
to make a `projects` directory in my home directory, and keep all my projects
there. Rust does not care where your code lives.

This actually leads to one other concern we should address: this guide will
assume that you have basic familiarity with the command line. Rust does not
require that you know a whole ton about the command line, but until the
language is in a more finished state, IDE support is spotty. Rust makes no
specific demands on your editing tooling, or where your code lives.

With that said, let's make a directory in our projects directory.

```{bash}
$ mkdir ~/projects
$ cd ~/projects
$ mkdir hello_world
$ cd hello_world
```

If you're on Windows and not using PowerShell, the `~` may not work. Consult
the documentation for your shell for more details.

Let's make a new source file next. I'm going to use the syntax `editor
filename` to represent editing a file in these examples, but you should use
whatever method you want. We'll call our file `main.rs`:

```{bash}
$ editor main.rs
```

Rust files always end in a `.rs` extension. If you're using more than one word
in your filename, use an underscore. `hello_world.rs` rather than
`helloworld.rs`.

Now that you've got your file open, type this in:

```{rust}
fn main() {
    println!("Hello, world!");
}
```

Save the file, and then type this into your terminal window:

```{bash}
$ rustc main.rs
$ ./main # or main.exe on Windows
Hello, world!
```

You can also run these examples on [play.rust-lang.org](http://play.rust-lang.org/) by clicking on the arrow that appears in the upper right of the example when you mouse over the code.

Success! Let's go over what just happened in detail.

```{rust}
fn main() {

}
```

These lines define a **function** in Rust. The `main` function is special:
it's the beginning of every Rust program. The first line says "I'm declaring a
function named `main`, which takes no arguments and returns nothing." If there
were arguments, they would go inside the parentheses (`(` and `)`), and because
we aren't returning anything from this function, we've dropped that notation
entirely.  We'll get to it later.

You'll also note that the function is wrapped in curly braces (`{` and `}`).
Rust requires these around all function bodies. It is also considered good
style to put the opening curly brace on the same line as the function
declaration, with one space in between.

Next up is this line:

```{rust}
    println!("Hello, world!");
```

This line does all of the work in our little program. There are a number of
details that are important here. The first is that it's indented with four
spaces, not tabs. Please configure your editor of choice to insert four spaces
with the tab key. We provide some [sample configurations for various
editors](https://github.com/rust-lang/rust/tree/master/src/etc).

The second point is the `println!()` part. This is calling a Rust **macro**,
which is how metaprogramming is done in Rust. If it were a function instead, it
would look like this: `println()`. For our purposes, we don't need to worry
about this difference. Just know that sometimes, you'll see a `!`, and that
means that you're calling a macro instead of a normal function. Rust implements
`println!` as a macro rather than a function for good reasons, but that's a
very advanced topic. You'll learn more when we talk about macros later. One
last thing to mention: Rust's macros are significantly different from C macros,
if you've used those. Don't be scared of using macros. We'll get to the details
eventually, you'll just have to trust us for now.

Next, `"Hello, world!"` is a **string**. Strings are a surprisingly complicated
topic in a systems programming language, and this is a **statically allocated**
string. We will talk more about different kinds of allocation later. We pass
this string as an argument to `println!`, which prints the string to the
screen. Easy enough!

Finally, the line ends with a semicolon (`;`). Rust is an **expression
oriented** language, which means that most things are expressions. The `;` is
used to indicate that this expression is over, and the next one is ready to
begin. Most lines of Rust code end with a `;`. We will cover this in-depth
later in the guide.

Finally, actually **compiling** and **running** our program. We can compile
with our compiler, `rustc`, by passing it the name of our source file:

```{bash}
$ rustc main.rs
```

This is similar to `gcc` or `clang`, if you come from a C or C++ background. Rust
will output a binary executable. You can see it with `ls`:

```{bash}
$ ls
main  main.rs
```

Or on Windows:

```{bash}
$ dir
main.exe  main.rs
```

There are now two files: our source code, with the `.rs` extension, and the
executable (`main.exe` on Windows, `main` everywhere else)

```{bash}
$ ./main  # or main.exe on Windows
```

This prints out our `Hello, world!` text to our terminal.

If you come from a dynamically typed language like Ruby, Python, or JavaScript,
you may not be used to these two steps being separate. Rust is an
**ahead-of-time compiled language**, which means that you can compile a
program, give it to someone else, and they don't need to have Rust installed.
If you give someone a `.rb` or `.py` or `.js` file, they need to have
Ruby/Python/JavaScript installed, but you just need one command to both compile
and run your program. Everything is a tradeoff in language design, and Rust has
made its choice.

Congratulations! You have officially written a Rust program. That makes you a
Rust programmer! Welcome.

Next, I'd like to introduce you to another tool, Cargo, which is used to write
real-world Rust programs. Just using `rustc` is nice for simple things, but as
your project grows, you'll want something to help you manage all of the options
that it has, and to make it easy to share your code with other people and
projects.
