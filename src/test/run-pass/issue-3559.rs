// rustc --test map_to_str.rs && ./map_to_str
extern mod std;
use io::{WriterUtil};
use std::map::*;

#[cfg(test)]
fn check_strs(actual: &str, expected: &str) -> bool
{
    if actual != expected
    {
        io::stderr().write_line(fmt!("Found %s, but expected %s", actual, expected));
        return false;
    }
    return true;
}

#[test]
fn tester()
{
    let table = HashMap();
    table.insert(@~"one", 1);
    table.insert(@~"two", 2);
    assert check_strs(table.to_str(), ~"xxx");   // not sure what expected should be
}

fn main() {}
