% Comments

Now that we have some functions, it's a good idea to learn about comments.
Comments are notes that you leave to other programmers to help explain things
about your code. The compiler mostly ignores them.

Rust has two kinds of comments that you should care about: **line comment**s
and **doc comment**s.

```{rust}
// Line comments are anything after '//' and extend to the end of the line.

let x = 5; // this is also a line comment.

// If you have a long explanation for something, you can put line comments next
// to each other. Put a space between the // and your comment so that it's
// more readable.
```

The other kind of comment is a doc comment. Doc comments use `///` instead of
`//`, and support Markdown notation inside:

```{rust}
/// `hello` is a function that prints a greeting that is personalized based on
/// the name given.
///
/// # Arguments
///
/// * `name` - The name of the person you'd like to greet.
///
/// # Example
///
/// ```rust
/// let name = "Steve";
/// hello(name); // prints "Hello, Steve!"
/// ```
fn hello(name: &str) {
    println!("Hello, {}!", name);
}
```

When writing doc comments, adding sections for any arguments, return values,
and providing some examples of usage is very, very helpful.

You can use the [`rustdoc`](../rustdoc.html) tool to generate HTML documentation
from these doc comments.
