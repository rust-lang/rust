% Standard Input

Getting input from the keyboard is pretty easy, but uses some things
we haven't seen before. Here's a simple program that reads some input,
and then prints it back out:

```{rust,ignore}
fn main() {
    println!("Type something!");

    let input = std::io::stdin().read_line().ok().expect("Failed to read line");

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
the `use` statement to import it in:

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

    let input = io::stdin().read_line().ok().expect("Failed to read line");

    println!("{}", input);
}
```

Next up:

```{rust,ignore}
.read_line()
```

The `read_line()` method can be called on the result of `stdin()` to return
a full line of input. Nice and easy.

```{rust,ignore}
.ok().expect("Failed to read line");
```

Do you remember this code?

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
        OptionalInt::Missing  => println!("x is missing!"),
    }

    match y {
        OptionalInt::Value(n) => println!("y is {}", n),
        OptionalInt::Missing  => println!("y is missing!"),
    }
}
```

We had to match each time to see if we had a value or not. In this case,
though, we _know_ that `x` has a `Value`, but `match` forces us to handle
the `missing` case. This is what we want 99% of the time, but sometimes, we
know better than the compiler.

Likewise, `read_line()` does not return a line of input. It _might_ return a
line of input, though it might also fail to do so. This could happen if our program
isn't running in a terminal, but as part of a cron job, or some other context
where there's no standard input. Because of this, `read_line` returns a type
very similar to our `OptionalInt`: an `IoResult<T>`. We haven't talked about
`IoResult<T>` yet because it is the **generic** form of our `OptionalInt`.
Until then, you can think of it as being the same thing, just for any type –
not just `i32`s.

Rust provides a method on these `IoResult<T>`s called `ok()`, which does the
same thing as our `match` statement but assumes that we have a valid value.
We then call `expect()` on the result, which will terminate our program if we
don't have a valid value. In this case, if we can't get input, our program
doesn't work, so we're okay with that. In most cases, we would want to handle
the error case explicitly. `expect()` allows us to give an error message if
this crash happens.

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

    let input = io::stdin()                       // std::io::stdio::StdinReader
                  .read_line()                    // IoResult<String>
                  .ok()                           // Option<String>
                  .expect("Failed to read line"); // String

    println!("{}", input);
}
```

Sometimes, this makes things more readable – sometimes, less. Use your judgement
here.

That's all you need to get basic input from the standard input! It's not too
complicated, but there are a number of small parts.
