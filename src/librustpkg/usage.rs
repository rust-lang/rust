use core::io;

pub fn general() {
    io::println(~"Usage: rustpkg [options] <cmd> [args..]

Where <cmd> is one of:
    build, clean, install, prefer, test, uninstall, unprefer

Options:

    -h, --help                  Display this message
    <cmd> -h, <cmd> --help      Display help for <cmd>");
}

pub fn build() {
    io::println(~"rustpkg [options..] build

Build all targets described in the package script in the current
directory.

Options:
    -c, --cfg      Pass a cfg flag to the package script");
}

pub fn clean() {
    io::println(~"rustpkg clean

Remove all build files in the work cache for the package in the current
directory.");
}

pub fn install() {
    io::println(~"rustpkg [options..] install [url] [target]

Install a package from a URL by Git or cURL (FTP, HTTP, etc.).
If target is provided, Git will checkout the branch or tag before
continuing. If the URL is a TAR file (with or without compression),
extract it before installing. If a URL isn't provided, the package will
be built and installed from the current directory (which is
functionally the same as `rustpkg build` and installing the result).

Examples:
    rustpkg install
    rustpkg install git://github.com/mozilla/servo.git
    rustpkg install git://github.com/mozilla/servo.git v0.1.2
    rustpkg install http://rust-lang.org/hello-world-0.3.4.tar.gz

Options:
    -c, --cfg      Pass a cfg flag to the package script
    -p, --prefer   Prefer the package after installing
                   (see `rustpkg prefer -h`)");
}

pub fn uninstall() {
    io::println(~"rustpkg uninstall <name>[@version]

Remove a package by name and/or version. If version is omitted then all
versions of the package will be removed. If the package[s] is/are depended
on by another package then they cannot be removed.  If the package is preferred
(see `rustpkg prefer -h`), it will attempt to prefer the next latest
version of the package if another version is installed, otherwise it'll remove
the symlink.");
}

pub fn prefer() {
    io::println(~"rustpkg [options..] prefer <name>[@version]

By default all binaries are given a unique name so that multiple versions can
coexist. The prefer command will symlink the uniquely named binary to
the binary directory under its bare name. The user will need to confirm
if the symlink will overwrite another. If version is not supplied, the latest
version of the package will be preferred.

Example:
    export PATH=$PATH:/home/user/.rustpkg/bin
    rustpkg prefer machine@1.2.4
    machine -v
    ==> v1.2.4
    rustpkg prefer machine@0.4.6
    machine -v
    ==> v0.4.6");
}

pub fn unprefer() {
    io::println(~"rustpkg [options..] unprefer <name>

Remove all symlinks from the store to the binary directory for a package
name. See `rustpkg prefer -h` for more information.");
}

pub fn test() {
    io::println(~"rustpkg [options..] test

Build all targets described in the package script in the current directory
with the test flag. The test bootstraps will be run afterwards and the output
and exit code will be redirected.

Options:
    -c, --cfg      Pass a cfg flag to the package script");
}
