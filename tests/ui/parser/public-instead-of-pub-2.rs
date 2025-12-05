// Checks what happens when `public` is used instead of the correct, `pub`
// Won't give help message for this case

public let x = 1;
//~^ ERROR expected one of `!` or `::`, found keyword `let`

fn main() { }
