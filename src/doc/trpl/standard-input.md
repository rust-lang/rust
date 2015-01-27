% Standard Input

Getting input from the keyboard is pretty easy, but uses some things
we haven't seen before. Here's a simple program that reads some input,
and then prints it back out:

```{rust,ignore}
fn main() {
    println!("Type something!");

    let result = std::io::stdin().read_line();

    let input = match result {
        Ok(text) => text,
        Err(_) => panic!("Failed to read line"),
    };

    println!("{}", input);
}
```

Let's go over these chunks, one by one:

```{rust,ignore}
std::io::stdin();
```

This calls a function, `stdin()`, that lives inside the `std::io` module. As
you can imagine, everything in `std` is provided by Rust, the 'standard
library.' We'll talk more about the module system later.

Since writing the fully qualified name all the time is annoying, we can use
the `use` statement to import it:

```{rust}
use std::io::stdin;

stdin();
```

However, it's considered better practice to not import individual functions, but
to import the module, and only use one level of qualification:

```{rust}
use std::io;

io::stdin();
```

Let's update our example to use this style:

```{rust,ignore}
use std::io;

fn main() {
    println!("Type something!");

    let result = io::stdin().read_line();

    let input = match result {
        Ok(text) => text,
        Err(_) => panic!("Failed to read line"),
    };

    println!("{}", input);
}
```

Next up:

```{rust,ignore}
.read_line()
```

The `read_line()` method can be called on the result of `stdin()` to return
a full line of input. Nice and easy.

It doesn't return a `String` though — not directly, at least. The returned value
is an `IoResult<T>`. Do you remember this code?

```{rust}
enum OptionalInt {
    Value(i32),
    Missing,
}

fn main() {
    let x = OptionalInt::Value(5);
    let y = OptionalInt::Missing;

    match x {
        OptionalInt::Value(n) => println!("x is {}", n),
        OptionalInt::Missing => println!("x is missing!"),
    }

    match y {
        OptionalInt::Value(n) => println!("y is {}", n),
        OptionalInt::Missing => println!("y is missing!"),
    }
}
```

The `OptionalInt` enum gave us a way to hold an integer if we had one, or
otherwise say that it's missing. `IoResult<T>` does a similar thing except it is
*generic*, which means it can be used for any type – not just `i32`s.

Instead of `Value` and `Missing`, `IoResult<T>`'s values are `Ok` and `Err`. An
`Ok` result means that the `read_line()` worked, and packed inside that `Ok` is
the text that came from standard input. An `Err` result means that it failed for
some reason and there is no text. This could happen if our program isn't running
in a terminal, but as part of a cron job, or some other context where there's no
standard input.

```{rust,ignore}
let input = match result {
    Ok(text) => text,
    Err(_) => panic!("Failed to read line"),
};
```

Next we use a `match` statement to see what type of result was returned. If the
`read_line()` was successful, it unpacks the value from inside the `Ok` and gives
it the name `text`. This text is then assigned to the `input` binding. If it was
unsuccessful, the `panic!()` macro terminates the program immediately with the
supplied error message.

Although this works, it would be annoying to need to write an entire `match`
statement every time we use `read_line()`. For dealing with this common
situation Rust has a more compact syntax which does the same thing. The
program can instead be written like this:

```{rust,ignore}
use std::io;

fn main() {
    println!("Type something!");

    let input = io::stdin().read_line().ok().expect("Failed to read line");

    println!("{}", input);
}
```

Every `IoResult<T>` has a method `ok()`. This attempts to extract the value from
inside the `Ok` – this was called `text` in the previous version. But this isn't
always going to work. If the `IoResult` is an `Err` type there is nothing to
return.

For this reason `ok()` doesn't return a `String` either. It returns an
`Option<T>`. This is an enum with two possible values – `Some` or `None`. `Some`
is a wrapper containing the value. `None` means the value does not exist. There
is one more step of unwrapping to do!

```{rust,ignore}
.expect("Failed to read line");
```

`Option<T>` provides a helpful method `expect()`. If it is a `Some<T>` type, it
will return the value inside it. This is the actual data the user typed, a
`String` in this case. If it's a `None` the program will terminate immediately,
printing the message passed in.

This version does exactly the same thing but requires much less typing!

We will cover the exact details of how all of this works later in the Guide.
For now, this gives you enough of a basic understanding to work with.

Back to the code we were working on! Here's a refresher:

```{rust,ignore}
use std::io;

fn main() {
    println!("Type something!");

    let input = io::stdin().read_line().ok().expect("Failed to read line");

    println!("{}", input);
}
```

With long lines like this, Rust gives you some flexibility with the whitespace.
We _could_ write the example like this:

```{rust,ignore}
use std::io;

fn main() {
    println!("Type something!");

    // here, we'll show the types at each step

    let input = io::stdin() // std::io::stdio::StdinReader
                  .read_line() // IoResult<String>
                  .ok() // Option<String>
                  .expect("Failed to read line"); // String

    println!("{}", input);
}
```

Sometimes, this makes things more readable – sometimes, less. Use your judgement
here.

That's all you need to get basic input from the standard input! It's not too
complicated, but there are a number of small parts.
