% Guessing Game

Okay! We've got the basics of Rust down. Let's write a bigger program.

For our first project, we'll implement a classic beginner programming problem:
the guessing game. Here's how it works: Our program will generate a random
integer between one and a hundred. It will then prompt us to enter a guess.
Upon entering our guess, it will tell us if we're too low or too high. Once we
guess correctly, it will congratulate us. Sound good?

## Set up

Let's set up a new project. Go to your projects directory. Remember how we
had to create our directory structure and a `Cargo.toml` for `hello_world`? Cargo
has a command that does that for us. Let's give it a shot:

```{bash}
$ cd ~/projects
$ cargo new guessing_game --bin
$ cd guessing_game
```

We pass the name of our project to `cargo new`, and then the `--bin` flag,
since we're making a binary, rather than a library.

Check out the generated `Cargo.toml`:

```toml
[package]

name = "guessing_game"
version = "0.0.1"
authors = ["Your Name <you@example.com>"]
```

Cargo gets this information from your environment. If it's not correct, go ahead
and fix that.

Finally, Cargo generated a "Hello, world!" for us. Check out `src/main.rs`:

```{rust}
fn main() {
    println!("Hello, world!")
}
```

Let's try compiling what Cargo gave us:

```{bash}
$ cargo build
   Compiling guessing_game v0.0.1 (file:///home/you/projects/guessing_game)
```

Excellent! Open up your `src/main.rs` again. We'll be writing all of
our code in this file. We'll talk about multiple-file projects later on in the
guide.

Before we move on, let me show you one more Cargo command: `run`. `cargo run`
is kind of like `cargo build`, but it also then runs the produced executable.
Try it out:

```bash
$ cargo run
   Compiling guessing_game v0.0.1 (file:///home/you/projects/guessing_game)
     Running `target/guessing_game`
Hello, world!
```

Great! The `run` command comes in handy when you need to rapidly iterate on a project.
Our game is just such a project, we need to quickly test each iteration before moving on to the next one.

## Processing a Guess

Let's get to it! The first thing we need to do for our guessing game is
allow our player to input a guess. Put this in your `src/main.rs`:

```{rust,no_run}
use std::io;

fn main() {
    println!("Guess the number!");

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");

    println!("You guessed: {}", input);
}
```

You've seen this code before, when we talked about standard input. We
import the `std::io` module with `use`, and then our `main` function contains
our program's logic. We print a little message announcing the game, ask the
user to input a guess, get their input, and then print it out.

Because we talked about this in the section on standard I/O, I won't go into
more details here. If you need a refresher, go re-read that section.

## Generating a secret number

Next, we need to generate a secret number. To do that, we need to use Rust's
random number generation, which we haven't talked about yet. Rust includes a
bunch of interesting functions in its standard library. If you need a bit of
code, it's possible that it's already been written for you! In this case,
we do know that Rust has random number generation, but we don't know how to
use it.

Enter the docs. Rust has a page specifically to document the standard library.
You can find that page [here](../std/index.html). There's a lot of information on
that page, but the best part is the search bar. Right up at the top, there's
a box that you can enter in a search term. The search is pretty primitive
right now, but is getting better all the time. If you type 'random' in that
box, the page will update to [this one](../std/index.html?search=random). The very
first result is a link to [`std::rand::random`](../std/rand/fn.random.html). If we
click on that result, we'll be taken to its documentation page.

This page shows us a few things: the type signature of the function, some
explanatory text, and then an example. Let's try to modify our code to add in the
`random` function and see what happens:

```{rust,ignore}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random() % 100) + 1; // secret_number: i32

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");


    println!("You guessed: {}", input);
}
```

The first thing we changed was to `use std::rand`, as the docs
explained.  We then added in a `let` expression to create a variable binding
named `secret_number`, and we printed out its result.

Also, you may wonder why we are using `%` on the result of `rand::random()`.
This operator is called 'modulo', and it returns the remainder of a division.
By taking the modulo of the result of `rand::random()`, we're limiting the
values to be between 0 and 99. Then, we add one to the result, making it from 1
to 100. Using modulo can give you a very, very small bias in the result, but
for this example, it is not important.

Let's try to compile this using `cargo build`:

```bash
$ cargo build
   Compiling guessing_game v0.0.1 (file:///home/you/projects/guessing_game)
src/main.rs:7:26: 7:34 error: the type of this value must be known in this context
src/main.rs:7     let secret_number = (rand::random() % 100) + 1;
                                       ^~~~~~~~
error: aborting due to previous error
```

It didn't work! Rust says "the type of this value must be known in this
context." What's up with that? Well, as it turns out, `rand::random()` can
generate many kinds of random values, not just integers. And in this case, Rust
isn't sure what kind of value `random()` should generate. So we have to help
it. With number literals, we can just add an `i32` onto the end to tell Rust they're
integers, but that does not work with functions. There's a different syntax,
and it looks like this:

```{rust,ignore}
rand::random::<i32>();
```

This says "please give me a random `i32` value." We can change our code to use
this hint:

```{rust,no_run}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<i32>() % 100) + 1;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");


    println!("You guessed: {}", input);
}
```

Try running our new program a few times:

```bash
$ cargo run
   Compiling guessing_game v0.0.1 (file:///home/you/projects/guessing_game)
     Running `target/guessing_game`
Guess the number!
The secret number is: 7
Please input your guess.
4
You guessed: 4
$ ./target/guessing_game
Guess the number!
The secret number is: 83
Please input your guess.
5
You guessed: 5
$ ./target/guessing_game
Guess the number!
The secret number is: -29
Please input your guess.
42
You guessed: 42
```

Wait. Negative 29? We wanted a number between one and a hundred! We have two
options here: we can either ask `random()` to generate an unsigned integer, which
can only be positive, or we can use the `abs()` function. Let's go with the
unsigned integer approach. If we want a random positive number, we should ask for
a random positive number. Our code looks like this now:

```{rust,no_run}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");


    println!("You guessed: {}", input);
}
```

And trying it out:

```bash
$ cargo run
   Compiling guessing_game v0.0.1 (file:///home/you/projects/guessing_game)
     Running `target/guessing_game`
Guess the number!
The secret number is: 57
Please input your guess.
3
You guessed: 3
```

Great! Next up: let's compare our guess to the secret guess.

## Comparing guesses

If you remember, earlier in the guide, we made a `cmp` function that compared
two numbers. Let's add that in, along with a `match` statement to compare our
guess to the secret number:

```{rust,ignore}
use std::io;
use std::rand;
use std::cmp::Ordering;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");


    println!("You guessed: {}", input);

    match cmp(input, secret_number) {
        Ordering::Less    => println!("Too small!"),
        Ordering::Greater => println!("Too big!"),
        Ordering::Equal   => println!("You win!"),
    }
}

fn cmp(a: i32, b: i32) -> Ordering {
    if a < b { Ordering::Less }
    else if a > b { Ordering::Greater }
    else { Ordering::Equal }
}
```

If we try to compile, we'll get some errors:

```bash
$ cargo build
   Compiling guessing_game v0.0.1 (file:///home/you/projects/guessing_game)
src/main.rs:20:15: 20:20 error: mismatched types: expected `i32` but found `collections::string::String` (expected i32 but found struct collections::string::String)
src/main.rs:20     match cmp(input, secret_number) {
                             ^~~~~
src/main.rs:20:22: 20:35 error: mismatched types: expected `i32` but found `uint` (expected i32 but found uint)
src/main.rs:20     match cmp(input, secret_number) {
                                    ^~~~~~~~~~~~~
error: aborting due to 2 previous errors
```

This often happens when writing Rust programs, and is one of Rust's greatest
strengths. You try out some code, see if it compiles, and Rust tells you that
you've done something wrong. In this case, our `cmp` function works on integers,
but we've given it unsigned integers. In this case, the fix is easy, because
we wrote the `cmp` function! Let's change it to take `uint`s:

```{rust,ignore}
use std::io;
use std::rand;
use std::cmp::Ordering;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");


    println!("You guessed: {}", input);

    match cmp(input, secret_number) {
        Ordering::Less    => println!("Too small!"),
        Ordering::Greater => println!("Too big!"),
        Ordering::Equal   => println!("You win!"),
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Ordering::Less }
    else if a > b { Ordering::Greater }
    else { Ordering::Equal }
}
```

And try compiling again:

```bash
$ cargo build
   Compiling guessing_game v0.0.1 (file:///home/you/projects/guessing_game)
src/main.rs:20:15: 20:20 error: mismatched types: expected `uint` but found `collections::string::String` (expected uint but found struct collections::string::String)
src/main.rs:20     match cmp(input, secret_number) {
                             ^~~~~
error: aborting due to previous error
```

This error is similar to the last one: we expected to get a `uint`, but we got
a `String` instead! That's because our `input` variable is coming from the
standard input, and you can guess anything. Try it:

```bash
$ ./target/guessing_game
Guess the number!
The secret number is: 73
Please input your guess.
hello
You guessed: hello
```

Oops! Also, you'll note that we just ran our program even though it didn't compile.
This works because the older version we did successfully compile was still lying
around. Gotta be careful!

Anyway, we have a `String`, but we need a `uint`. What to do? Well, there's
a function for that:

```{rust,ignore}
let input = io::stdin().read_line()
                       .ok()
                       .expect("Failed to read line");
let input_num: Option<uint> = input.parse();
```

The `parse` function takes in a `&str` value and converts it into something.
We tell it what kind of something with a type hint. Remember our type hint with
`random()`? It looked like this:

```{rust,ignore}
rand::random::<uint>();
```

There's an alternate way of providing a hint too, and that's declaring the type
in a `let`:

```{rust,ignore}
let x: uint = rand::random();
```

In this case, we say `x` is a `uint` explicitly, so Rust is able to properly
tell `random()` what to generate. In a similar fashion, both of these work:

```{rust,ignore}
let input_num = "5".parse::<uint>();         // input_num: Option<uint>
let input_num: Option<uint> = "5".parse();   // input_num: Option<uint>
```

Anyway, with us now converting our input to a number, our code looks like this:

```{rust,ignore}
use std::io;
use std::rand;
use std::cmp::Ordering;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");
    let input_num: Option<uint> = input.parse();

    println!("You guessed: {}", input_num);

    match cmp(input_num, secret_number) {
        Ordering::Less    => println!("Too small!"),
        Ordering::Greater => println!("Too big!"),
        Ordering::Equal   => println!("You win!"),
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Ordering::Less }
    else if a > b { Ordering::Greater }
    else { Ordering::Equal }
}
```

Let's try it out!

```bash
$ cargo build
   Compiling guessing_game v0.0.1 (file:///home/you/projects/guessing_game)
src/main.rs:22:15: 22:24 error: mismatched types: expected `uint` but found `core::option::Option<uint>` (expected uint but found enum core::option::Option)
src/main.rs:22     match cmp(input_num, secret_number) {
                             ^~~~~~~~~
error: aborting due to previous error
```

Oh yeah! Our `input_num` has the type `Option<uint>`, rather than `uint`. We
need to unwrap the Option. If you remember from before, `match` is a great way
to do that. Try this code:

```{rust,no_run}
use std::io;
use std::rand;
use std::cmp::Ordering;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");
    let input_num: Option<uint> = input.parse();

    let num = match input_num {
        Some(num) => num,
        None      => {
            println!("Please input a number!");
            return;
        }
    };


    println!("You guessed: {}", num);

    match cmp(num, secret_number) {
        Ordering::Less    => println!("Too small!"),
        Ordering::Greater => println!("Too big!"),
        Ordering::Equal   => println!("You win!"),
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Ordering::Less }
    else if a > b { Ordering::Greater }
    else { Ordering::Equal }
}
```

We use a `match` to either give us the `uint` inside of the `Option`, or else
print an error message and return. Let's give this a shot:

```bash
$ cargo run
   Compiling guessing_game v0.0.1 (file:///home/you/projects/guessing_game)
     Running `target/guessing_game`
Guess the number!
The secret number is: 17
Please input your guess.
5
Please input a number!
```

Uh, what? But we did!

... actually, we didn't. See, when you get a line of input from `stdin()`,
you get all the input. Including the `\n` character from you pressing Enter.
Therefore, `parse()` sees the string `"5\n"` and says "nope, that's not a
number; there's non-number stuff in there!" Luckily for us, `&str`s have an easy
method we can use defined on them: `trim()`. One small modification, and our
code looks like this:

```{rust,no_run}
use std::io;
use std::rand;
use std::cmp::Ordering;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");
    let input_num: Option<uint> = input.trim().parse();

    let num = match input_num {
        Some(num) => num,
        None      => {
            println!("Please input a number!");
            return;
        }
    };


    println!("You guessed: {}", num);

    match cmp(num, secret_number) {
        Ordering::Less    => println!("Too small!"),
        Ordering::Greater => println!("Too big!"),
        Ordering::Equal   => println!("You win!"),
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Ordering::Less }
    else if a > b { Ordering::Greater }
    else { Ordering::Equal }
}
```

Let's try it!

```bash
$ cargo run
   Compiling guessing_game v0.0.1 (file:///home/you/projects/guessing_game)
     Running `target/guessing_game`
Guess the number!
The secret number is: 58
Please input your guess.
  76
You guessed: 76
Too big!
```

Nice! You can see I even added spaces before my guess, and it still figured
out that I guessed 76. Run the program a few times, and verify that guessing
the number works, as well as guessing a number too small.

The Rust compiler helped us out quite a bit there! This technique is called
"lean on the compiler", and it's often useful when working on some code. Let
the error messages help guide you towards the correct types.

Now we've got most of the game working, but we can only make one guess. Let's
change that by adding loops!

## Looping

As we already discussed, the `loop` keyword gives us an infinite loop.
Let's add that in:

```{rust,no_run}
use std::io;
use std::rand;
use std::cmp::Ordering;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    loop {

        println!("Please input your guess.");

        let input = io::stdin().read_line()
                               .ok()
                               .expect("Failed to read line");
        let input_num: Option<uint> = input.trim().parse();

        let num = match input_num {
            Some(num) => num,
            None      => {
                println!("Please input a number!");
                return;
            }
        };


        println!("You guessed: {}", num);

        match cmp(num, secret_number) {
            Ordering::Less    => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal   => println!("You win!"),
        }
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Ordering::Less }
    else if a > b { Ordering::Greater }
    else { Ordering::Equal }
}
```

And try it out. But wait, didn't we just add an infinite loop? Yup. Remember
that `return`? If we give a non-number answer, we'll `return` and quit. Observe:

```bash
$ cargo run
   Compiling guessing_game v0.0.1 (file:///home/you/projects/guessing_game)
     Running `target/guessing_game`
Guess the number!
The secret number is: 59
Please input your guess.
45
You guessed: 45
Too small!
Please input your guess.
60
You guessed: 60
Too big!
Please input your guess.
59
You guessed: 59
You win!
Please input your guess.
quit
Please input a number!
```

Ha! `quit` actually quits. As does any other non-number input. Well, this is
suboptimal to say the least. First, let's actually quit when you win the game:

```{rust,no_run}
use std::io;
use std::rand;
use std::cmp::Ordering;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    loop {

        println!("Please input your guess.");

        let input = io::stdin().read_line()
                               .ok()
                               .expect("Failed to read line");
        let input_num: Option<uint> = input.trim().parse();

        let num = match input_num {
            Some(num) => num,
            None      => {
                println!("Please input a number!");
                return;
            }
        };


        println!("You guessed: {}", num);

        match cmp(num, secret_number) {
            Ordering::Less    => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal   => {
                println!("You win!");
                return;
            },
        }
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Ordering::Less }
    else if a > b { Ordering::Greater }
    else { Ordering::Equal }
}
```

By adding the `return` line after the `You win!`, we'll exit the program when
we win. We have just one more tweak to make: when someone inputs a non-number,
we don't want to quit, we just want to ignore it. Change that `return` to
`continue`:


```{rust,no_run}
use std::io;
use std::rand;
use std::cmp::Ordering;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    loop {

        println!("Please input your guess.");

        let input = io::stdin().read_line()
                               .ok()
                               .expect("Failed to read line");
        let input_num: Option<uint> = input.trim().parse();

        let num = match input_num {
            Some(num) => num,
            None      => {
                println!("Please input a number!");
                continue;
            }
        };


        println!("You guessed: {}", num);

        match cmp(num, secret_number) {
            Ordering::Less    => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal   => {
                println!("You win!");
                return;
            },
        }
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Ordering::Less }
    else if a > b { Ordering::Greater }
    else { Ordering::Equal }
}
```

Now we should be good! Let's try:

```bash
$ cargo run
   Compiling guessing_game v0.0.1 (file:///home/you/projects/guessing_game)
     Running `target/guessing_game`
Guess the number!
The secret number is: 61
Please input your guess.
10
You guessed: 10
Too small!
Please input your guess.
99
You guessed: 99
Too big!
Please input your guess.
foo
Please input a number!
Please input your guess.
61
You guessed: 61
You win!
```

Awesome! With one tiny last tweak, we have finished the guessing game. Can you
think of what it is? That's right, we don't want to print out the secret number.
It was good for testing, but it kind of ruins the game. Here's our final source:

```{rust,no_run}
use std::io;
use std::rand;
use std::cmp::Ordering;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    loop {

        println!("Please input your guess.");

        let input = io::stdin().read_line()
                               .ok()
                               .expect("Failed to read line");
        let input_num: Option<uint> = input.trim().parse();

        let num = match input_num {
            Some(num) => num,
            None      => {
                println!("Please input a number!");
                continue;
            }
        };


        println!("You guessed: {}", num);

        match cmp(num, secret_number) {
            Ordering::Less    => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal   => {
                println!("You win!");
                return;
            },
        }
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Ordering::Less }
    else if a > b { Ordering::Greater }
    else { Ordering::Equal }
}
```

## Complete!

At this point, you have successfully built the Guessing Game! Congratulations!

You've now learned the basic syntax of Rust. All of this is relatively close to
various other programming languages you have used in the past. These
fundamental syntactical and semantic elements will form the foundation for the
rest of your Rust education.

Now that you're an expert at the basics, it's time to learn about some of
Rust's more unique features.
