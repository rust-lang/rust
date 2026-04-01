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

fn main() {}
