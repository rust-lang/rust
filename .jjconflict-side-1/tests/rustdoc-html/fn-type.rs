#![crate_name = "foo"]
#![crate_type = "lib"]

pub struct Foo<'a, T> {
    pub generic: fn(val: &T) -> T,

    pub lifetime: fn(val: &'a i32) -> i32,
    pub hrtb_lifetime: for<'b, 'c> fn(one: &'b i32, two: &'c &'b i32) -> (&'b i32, &'c i32),
}

//@ has 'foo/struct.Foo.html' '//span[@id="structfield.generic"]' "generic: fn(val: &T) -> T"
//@ has 'foo/struct.Foo.html' '//span[@id="structfield.lifetime"]' "lifetime: fn(val: &'a i32) -> i32"
//@ has 'foo/struct.Foo.html' '//span[@id="structfield.hrtb_lifetime"]' "hrtb_lifetime: for<'b, 'c> fn(one: &'b i32, two: &'c &'b i32) -> (&'b i32, &'c i32)"
