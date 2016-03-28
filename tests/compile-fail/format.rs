#![feature(plugin)]
#![plugin(clippy)]
#![deny(useless_format)]

fn main() {
    format!("foo"); //~ERROR useless use of `format!`

    format!("{}", "foo"); //~ERROR useless use of `format!`
    format!("{:?}", "foo"); // we only want to warn about `{}`
    format!("{:+}", "foo"); // we only want to warn about `{}`
    format!("foo {}", "bar");
    format!("{} bar", "foo");

    let arg: String = "".to_owned();
    format!("{}", arg); //~ERROR useless use of `format!`
    format!("{:?}", arg); // we only want to warn about `{}`
    format!("{:+}", arg); // we only want to warn about `{}`
    format!("foo {}", arg);
    format!("{} bar", arg);

    // we don’t want to warn for non-string args, see #697
    format!("{}", 42);
    format!("{:?}", 42);
    format!("{:+}", 42);
    format!("foo {}", 42);
    format!("{} bar", 42);

    // we only want to warn about `format!` itself
    println!("foo");
    println!("{}", "foo");
    println!("foo {}", "foo");
    println!("{}", 42);
    println!("foo {}", 42);
}
