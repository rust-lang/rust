fn bot<T>() -> T { loop {} }

fn mutate(s: &mut str) {
    s[1..2] = bot();
    //~^ ERROR the size for values of type
    //~| ERROR the size for values of type
    s[1usize] = bot();
    //~^ ERROR the type `str` cannot be mutably indexed by `usize`
}

pub fn main() {}
