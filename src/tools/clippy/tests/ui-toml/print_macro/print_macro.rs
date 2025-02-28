//@compile-flags: --test
#![warn(clippy::print_stdout)]
#![warn(clippy::print_stderr)]

fn foo(n: u32) {
    print!("{n}");
    //~^ print_stdout
    eprint!("{n}");
    //~^ print_stderr
}

#[test]
pub fn foo1() {
    print!("{}", 1);
    eprint!("{}", 1);
}

#[cfg(test)]
fn foo3() {
    print!("{}", 1);
    eprint!("{}", 1);
}
