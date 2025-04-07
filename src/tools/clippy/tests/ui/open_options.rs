#![allow(unused_must_use)]
#![warn(clippy::nonsensical_open_options)]

use std::fs::OpenOptions;

trait OpenOptionsExt {
    fn truncate_write(&mut self, opt: bool) -> &mut Self;
}

impl OpenOptionsExt for OpenOptions {
    fn truncate_write(&mut self, opt: bool) -> &mut Self {
        self.truncate(opt).write(opt)
    }
}

fn main() {
    OpenOptions::new().read(true).truncate(true).open("foo.txt");
    //~^ nonsensical_open_options

    OpenOptions::new().append(true).truncate(true).open("foo.txt");
    //~^ nonsensical_open_options

    OpenOptions::new().read(true).read(false).open("foo.txt");
    //~^ nonsensical_open_options

    OpenOptions::new()
        .create(true)
        .truncate(true) // Ensure we don't trigger suspicious open options by having create without truncate
        .create(false)
        //~^ nonsensical_open_options
        .open("foo.txt");
    OpenOptions::new().write(true).write(false).open("foo.txt");
    //~^ nonsensical_open_options

    OpenOptions::new().append(true).append(false).open("foo.txt");
    //~^ nonsensical_open_options

    OpenOptions::new().truncate(true).truncate(false).open("foo.txt");
    //~^ nonsensical_open_options

    std::fs::File::options().read(true).read(false).open("foo.txt");
    //~^ nonsensical_open_options

    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    options.read(false);
    // Possible future improvement: recognize open options method call chains across statements.
    options.open("foo.txt");

    let mut options = std::fs::OpenOptions::new();
    options.truncate(true);
    options.create(true).open("foo.txt");

    OpenOptions::new().create(true).truncate_write(true).open("foo.txt");
}
