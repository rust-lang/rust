fn underscore_lifetime<'_>(str1: &'_ str, str2: &'_ str) -> &'_ str {
    //~^ ERROR: `'_` cannot be used here [E0637]
    //~| ERROR: missing lifetime specifier
    if str1.len() > str2.len() {
        str1
    } else {
        str2
    }
}

fn and_without_explicit_lifetime<T>()
where
    T: Into<&u32>, //~ ERROR: `&` without an explicit lifetime name cannot be used here [E0637]
{
}

// Regression test for https://github.com/rust-lang/rust/issues/156456.
fn associated_type_binding<I: IntoIterator<Item = &String>>() {}
//~^ ERROR: `&` without an explicit lifetime name cannot be used here [E0637]

trait T {
    type Assoc;
}

// Regression test for https://github.com/rust-lang/rust/issues/122025.
fn foo<F>(t: F) where F: T<Assoc=&str> {}
//~^ ERROR: `&` without an explicit lifetime name cannot be used here [E0637]

// Regression test for https://github.com/rust-lang/rust/issues/123713.
fn one_assoc<'a,F>(t: &'a F) where &'a F: T<Assoc=&str> {}// suggest &'a str
//~^ ERROR E0637
fn one<'a,F>(t: &'a F) where F:Into<&u32> {}// suggest to use 'a or for<'b>
//~^ ERROR E0637
fn multiple_assoc<'a,'b,F>(t: &'a F) where &'a F: T<Assoc=&str>{}
//~^ ERROR E0637
// suggest the user to choose one of the available lifetimes
fn multiple<'a,'b,F>(t: &'a F) where F:Into<&u32>{}
//~^ ERROR E0637
// suggest the user to choose one of the available lifetimes or for<'c>
fn main() {}
