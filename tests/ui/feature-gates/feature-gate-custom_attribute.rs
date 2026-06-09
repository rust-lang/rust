// Check that literals in attributes parse just fine.

#[fake_attr] //~ ERROR cannot find attribute `fake_attr` in this scope
#[fake_attr(100)] //~ ERROR cannot find attribute `fake_attr` in this scope
#[fake_attr(1, 2, 3)] //~ ERROR cannot find attribute `fake_attr` in this scope
#[fake_attr("hello")] //~ ERROR cannot find attribute `fake_attr` in this scope
#[fake_attr(name = "hello")] //~ ERROR cannot find attribute `fake_attr` in this scope
#[fake_attr(1, "hi", key = 12, true, false)] //~ ERROR cannot find attribute `fake_attr` in th
#[fake_attr(key = "hello", val = 10)] //~ ERROR cannot find attribute `fake_attr` in this scop
#[fake_attr(key("hello"), val(10))] //~ ERROR cannot find attribute `fake_attr` in this scope
#[fake_attr(enabled = true, disabled = false)] //~ ERROR cannot find attribute `fake_attr` in
#[fake_attr(true)] //~ ERROR cannot find attribute `fake_attr` in this scope
#[fake_attr(pi = 3.14159)] //~ ERROR cannot find attribute `fake_attr` in this scope
#[fake_attr(b"hi")] //~ ERROR cannot find attribute `fake_attr` in this scope
#[fake_doc(r"doc")] //~ ERROR cannot find attribute `fake_doc` in this scope
struct Q {}

fn main() {}
