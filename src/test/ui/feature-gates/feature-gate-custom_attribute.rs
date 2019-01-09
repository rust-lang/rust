// Check that literals in attributes parse just fine.


#![allow(dead_code)]
#![allow(unused_variables)]

#[fake_attr] //~ ERROR attribute `fake_attr` is currently unknown
#[fake_attr(100)] //~ ERROR attribute `fake_attr` is currently unknown
#[fake_attr(1, 2, 3)] //~ ERROR attribute `fake_attr` is currently unknown
#[fake_attr("hello")] //~ ERROR attribute `fake_attr` is currently unknown
#[fake_attr(name = "hello")] //~ ERROR attribute `fake_attr` is currently unknown
#[fake_attr(1, "hi", key = 12, true, false)] //~ ERROR attribute `fake_attr` is currently unknown
#[fake_attr(key = "hello", val = 10)] //~ ERROR attribute `fake_attr` is currently unknown
#[fake_attr(key("hello"), val(10))] //~ ERROR attribute `fake_attr` is currently unknown
#[fake_attr(enabled = true, disabled = false)] //~ ERROR attribute `fake_attr` is currently unknown
#[fake_attr(true)] //~ ERROR attribute `fake_attr` is currently unknown
#[fake_attr(pi = 3.14159)] //~ ERROR attribute `fake_attr` is currently unknown
#[fake_attr(b"hi")] //~ ERROR attribute `fake_attr` is currently unknown
#[fake_doc(r"doc")] //~ ERROR attribute `fake_doc` is currently unknown
struct Q {  }


fn main() { }
