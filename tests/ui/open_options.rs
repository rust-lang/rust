use std::fs::OpenOptions;

#[allow(unused_must_use)]
#[warn(clippy::nonsensical_open_options)]
fn main() {
    OpenOptions::new().read(true).truncate(true).open("foo.txt");
    //~^ ERROR: file opened with `truncate` and `read`
    //~| NOTE: `-D clippy::nonsensical-open-options` implied by `-D warnings`
    OpenOptions::new().append(true).truncate(true).open("foo.txt");
    //~^ ERROR: file opened with `append` and `truncate`

    OpenOptions::new().read(true).read(false).open("foo.txt");
    //~^ ERROR: the method `read` is called more than once
    OpenOptions::new().create(true).create(false).open("foo.txt");
    //~^ ERROR: the method `create` is called more than once
    OpenOptions::new().write(true).write(false).open("foo.txt");
    //~^ ERROR: the method `write` is called more than once
    OpenOptions::new().append(true).append(false).open("foo.txt");
    //~^ ERROR: the method `append` is called more than once
    OpenOptions::new().truncate(true).truncate(false).open("foo.txt");
    //~^ ERROR: the method `truncate` is called more than once
}
