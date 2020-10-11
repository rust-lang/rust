fn bot<T>() -> T { loop {} }

fn mutate(s: &mut str) {
    s[1..2] = bot(); //~ ERROR the size for values of type
    // A second error would happen due to `s[1..2]` but it isn't emitted because it is delayed and
    // there are already other errors being emitted.
    s[1usize] = bot(); //~ ERROR the type `str` cannot be indexed by `usize`
    s.get_mut(1); //~ ERROR the type `str` cannot be indexed by `{integer}`
    s.get_unchecked_mut(1); //~ ERROR the type `str` cannot be indexed by `{integer}`
    s['c']; //~ ERROR the type `str` cannot be indexed by `char`
}

pub fn main() {}
