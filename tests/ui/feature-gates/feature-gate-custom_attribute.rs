// Check that literals in attributes parse just fine.

#[fake_attr] //~ ERROR cannot find attribute `fake_attr`
#[fake_attr(100)] //~ ERROR cannot find attribute `fake_attr`
#[fake_attr(1, 2, 3)] //~ ERROR cannot find attribute `fake_attr`
#[fake_attr("hello")] //~ ERROR cannot find attribute `fake_attr`
#[fake_attr(name = "hello")] //~ ERROR cannot find attribute `fake_attr`
#[fake_attr(1, "hi", key = 12, true, false)] //~ ERROR cannot find attribute `fake_attr`
#[fake_attr(key = "hello", val = 10)] //~ ERROR cannot find attribute `fake_attr`
#[fake_attr(key("hello"), val(10))] //~ ERROR cannot find attribute `fake_attr`
#[fake_attr(enabled = true, disabled = false)] //~ ERROR cannot find attribute `fake_attr`
#[fake_attr(true)] //~ ERROR cannot find attribute `fake_attr`
#[fake_attr(pi = 3.14159)] //~ ERROR cannot find attribute `fake_attr`
#[fake_attr(b"hi")] //~ ERROR cannot find attribute `fake_attr`
#[fake_doc(r"doc")] //~ ERROR cannot find attribute `fake_doc`
struct Q {}

fn main() {}
