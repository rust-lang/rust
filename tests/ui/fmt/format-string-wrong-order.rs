fn main() {
    let bar = 3;
    format!("{?:}", bar);
    //~^ ERROR invalid format string: expected format parameter to occur after `:`
    format!("{?:bar}");
    //~^ ERROR invalid format string: expected format parameter to occur after `:`
    format!("{?:?}", bar);
    //~^ ERROR invalid format string: expected format parameter to occur after `:`
    format!("{??}", bar);
    //~^ ERROR invalid format string: expected `'}'`, found `'?'`
    format!("{?;bar}");
    //~^ ERROR invalid format string: expected `'}'`, found `'?'`
    format!("{?:#?}", bar);
    //~^ ERROR invalid format string: expected format parameter to occur after `:`
}
