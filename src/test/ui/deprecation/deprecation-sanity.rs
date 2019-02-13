// Various checks that deprecation attributes are used correctly

mod bogus_attribute_types_1 {
    #[deprecated(since = "a", note = "a", reason)] //~ ERROR unknown meta item 'reason'
    fn f1() { }

    #[deprecated(since = "a", note)] //~ ERROR incorrect meta item
    fn f2() { }

    #[deprecated(since, note = "a")] //~ ERROR incorrect meta item
    fn f3() { }

    #[deprecated(since = "a", note(b))] //~ ERROR incorrect meta item
    fn f5() { }

    #[deprecated(since(b), note = "a")] //~ ERROR incorrect meta item
    fn f6() { }

    #[deprecated(note = b"test")] //~ ERROR literal in `deprecated` value must be a string
    fn f7() { }

    #[deprecated("test")] //~ ERROR item in `deprecated` must be a key/value pair
    fn f8() { }
}

#[deprecated(since = "a", note = "b")]
#[deprecated(since = "a", note = "b")]
fn multiple1() { } //~ ERROR multiple deprecated attributes

#[deprecated(since = "a", since = "b", note = "c")] //~ ERROR multiple 'since' items
fn f1() { }

fn main() { }
