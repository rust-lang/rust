/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

fn call_bare(f: fn(&str)) {
    f("Hello ");
}

fn main() {
    let string = "world!";
    let f: |&str| = |s| println(s + string);
    call_bare(f)    //~ ERROR mismatched types
}

