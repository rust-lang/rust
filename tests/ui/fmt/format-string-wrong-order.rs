fn main() {
    let bar = 3;
    format!("{?:}", bar);
    //~^ ERROR invalid format string: expected format parameter to occur after `:`
    format!("{?:bar}");
    //~^ ERROR invalid format string: expected format parameter to occur after `:`
    format!("{?:?}", bar);
    //~^ ERROR invalid format string: expected format parameter to occur after `:`
    format!("{??}", bar);
    //~^ ERROR invalid format string: expected `}`, found `?`
    format!("{?;bar}");
    //~^ ERROR invalid format string: expected `}`, found `?`
    format!("{?:#?}", bar);
    //~^ ERROR invalid format string: expected format parameter to occur after `:`
    format!("Hello {<5:}!", "x");
    //~^ ERROR invalid format string: expected alignment specifier after `:` in format string; example: `{:>?}`
    format!("Hello {^5:}!", "x");
    //~^ ERROR invalid format string: expected alignment specifier after `:` in format string; example: `{:>?}`
    format!("Hello {>5:}!", "x");
    //~^ ERROR invalid format string: expected alignment specifier after `:` in format string; example: `{:>?}`
    println!("{0:#X>18}", 12345);
    //~^ ERROR invalid format string: expected alignment specifier after `:` in format string; example: `{:>?}`
    format!("{+:}");
    //~^ ERROR invalid format string: the `+` sign flag must appear after `:` in a format string
    format!("{bar+:}");
    //~^ ERROR invalid format string: the `+` sign flag must appear after `:` in a format string
}
