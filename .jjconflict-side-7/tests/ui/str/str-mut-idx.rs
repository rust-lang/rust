fn bot<T>() -> T { loop {} }

fn mutate(s: &mut str) {
    s[1..2] = bot();
    //~^ ERROR the size for values of type
    //~| ERROR the size for values of type
    s[1usize] = bot();
    //~^ ERROR the type `str` cannot be indexed by `usize`
    s.get_mut(1);
    //~^ ERROR the type `str` cannot be indexed by `{integer}`
    s.get_unchecked_mut(1);
    //~^ ERROR the type `str` cannot be indexed by `{integer}`
    s['c'];
    //~^ ERROR the type `str` cannot be indexed by `char`
}

pub fn main() {}
