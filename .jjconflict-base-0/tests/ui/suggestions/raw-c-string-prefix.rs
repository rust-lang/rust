// `rc` and `cr` are easy to confuse; check that we issue a suggestion to help.

//@ edition:2021

fn main() {
    rc"abc";
    //~^ ERROR: prefix `rc` is unknown
    //~| HELP: use `cr` for a raw C-string
    //~| ERROR: expected one of
}
