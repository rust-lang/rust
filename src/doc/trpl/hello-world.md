% Hello, world!

Now that you have Rust installed, we'll help you write your first Rust program.
It's traditional when learning a new language to write a little program to
print the text “Hello, world!” to the screen, and in this section, we'll follow
that tradition. 

The nice thing about starting with such a simple program is that you can
quickly verify that your compiler is installed, and that it's working properly.
Printing information to the screen is also just a pretty common thing to do, so
practicing it early on is good.

> Note: This book assumes basic familiarity with the command line. Rust itself
> makes no specific demands about your editing, tooling, or where your code
> lives, so if you prefer an IDE to the command line, that's an option. You may
> want to check out [SolidOak], which was built specifically with Rust in mind.
> There are a number of extensions in development by the community, and the
> Rust team ships plugins for [various editors]. Configuring your editor or
> IDE is out of the scope of this tutorial, so check the documentation for your
> specific setup. 

[SolidOak]: https://github.com/oakes/SolidOak
[various editors]: https://github.com/rust-lang/rust/blob/master/src/etc/CONFIGS.md

## Creating a Project File

First, make a file to put your Rust code in. Rust doesn't care where your code
lives, but for this book, I suggest making a *projects* directory in your home
directory, and keeping all your projects there. Open a terminal and enter the
following commands to make a directory for this particular project:

```bash
$ mkdir ~/projects
$ cd ~/projects
$ mkdir hello_world
$ cd hello_world
```

> Note: If you’re on Windows and not using PowerShell, the `~` may not work.
> Consult the documentation for your shell for more details.

## Writing and Running a Rust Program

Next, make a new source file next and call it *main.rs*. Rust files always end
in a *.rs* extension. If you’re using more than one word in your filename, use
an underscore to separate them; for example, you'd use *hello_world.rs* rather
than *helloworld.rs*.

Now open the *main.rs* file you just created, and type the following code:

```rust
fn main() {
    println!("Hello, world!");
}
```

Save the file, and go back to your terminal window. On Linux or OSX, enter the
following commands:

```bash
$ rustc main.rs
$ ./main 
Hello, world!
```

In Windows, just replace `main` with `main.exe`. Regardless of your operating
system, you should see the string `Hello, world!` print to the terminal. If you
did, then congratulations! You've officially written a Rust program. That makes
you a Rust programmer! Welcome. 

#Anatomy of a Rust Program

Now, let’s go over what just happened in your "Hello, world!" program in
detail. Here's the first piece of the puzzle:

```rust
fn main() {

}
```

These lines define a *function* in Rust. The `main` function is special: it's
the beginning of every Rust program. The first line says, "I’m declaring a
function named `main` that currently takes no arguments and returns nothing."
If there were arguments, they would go inside the parentheses (`(` and `)`),
and because we aren’t returning anything from this function, we can omit the
return type entirely.

Also note that the function body is wrapped in curly braces (`{` and `}`). Rust
requires these around all function bodies. It's considered good style to put
the opening curly brace on the same line as the function declaration, with one
line space in between.

Inside the `main()` function, is this line:

```rust
    println!("Hello, world!");
```

This line does all of the work in this little program: it prints text to the
screen. There are a number of details that are important here. The first is
that it’s indented with four spaces, not tabs. If you configure your editor of
choice to insert four spaces with the tab key, it will make your coding much
more efficient.
 
The second important part is the `println!()` line. This is calling a Rust
*[macro]*, which is how metaprogramming is done in Rust. If it were calling a
function instead, it would look like this: `println()` (without the !). We'll
discuss Rust macros in more detail in Chapter XX, but for now you just need to
know that when you see a `!` that means that you’re calling a macro instead of
a normal function. 


[macro]: macros.html

Next is `"Hello, world!"` which is a *string*. Strings are a surprisingly
complicated topic in a systems programming language, and this is a *[statically
allocated]* string. We pass this string as an argument to `println!`, which
prints the string to the screen. Easy enough!

[allocation]: the-stack-and-the-heap.html

The line ends with a semicolon (`;`). Rust is an *[expression oriented]*
language, which means that most things are expressions, rather than statements.
The `;` indicates that this expression is over, and the next one is ready to
begin. Most lines of Rust code end with a `;`.

[expression-oriented language]: glossary.html#expression-oriented-language

# Compiling and Running Are Separate Steps

In "Writing and Running a Rust Program", we showed you how to run a newly
created program. We'll break that process down and examine each step now. 

Before running a Rust program, you have to compile it. You can use the Rust
compiler by entering the `rustc` command and passing it the name of your source
file, like this:

```bash
$ rustc main.rs
```

If you come from a C or C++ background, you'll notice that this is similar to
`gcc` or `clang`. After compiling successfully, Rust should output a binary
executable, which you can see on Linux or OSX by entering the `ls` command in
your shell as follows:

```bash
$ ls
main  main.rs
```

On Windows, you'd enter:

```bash
$ dir
main.exe  main.rs
```

This would create two files: the source code, with a `.rs` extension, and the
executable (`main.exe` on Windows, `main` everywhere else). All that's left to
do from here is run the `main` or `main.exe` file, like this:

```bash
$ ./main  # or main.exe on Windows
```

If *main.rs* were your "Hello, world!" program, this would print `Hello,
world!` to your terminal.

If you come from a dynamic language like Ruby, Python, or JavaScript, you may
not be used to compiling and running a program being separate steps. Rust is an
*ahead-of-time compiled* language, which means that you can compile a program,
give it to someone else, and they can run it even without Rust installed. If
you give someone a `.rb` or `.py` or `.js` file, on the other hand, they need
to have a Ruby, Python, or JavaScript implementation installed (respectively),
but you only need one command to both compile and run your program. Everything
is a tradeoff in language design, and Rust has made its choice.

Just compiling with `rustc` is fine for simple programs, but as your project
grows, you'll want to be able to manage all of the options your project has,
and make it easy to share your code with other people and projects. Next, I'll
introduce you to a tool called Cargo, which will help you write real-world Rust
programs.

