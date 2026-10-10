trait Bar {}
safe impl Bar for () { }
//~^ ERROR expected one of `!` or `::`, found keyword `impl`

fn main() {}
