pub fn will_be_called() -> &'static str {
    let val = "called";
    println!("{}", val);
    val
}

pub fn will_not_be_called() -> bool {
    println!("should not have been called");
    false
}

pub fn print<T>(left: &str, value: T, right: &str)
where
    T: std::fmt::Display,
{
    println!("{}{}{}", left, value, right);
}

pub fn wrap_with<F, T>(inner: T, should_wrap: bool, wrapper: F)
where
    F: FnOnce(&T)
{
    if should_wrap {
        wrapper(&inner)
    }
}

fn main() {
    let less = 1;
    let more = 100;

    if less < more {
        wrap_with(will_be_called(), less < more, |inner| print(" ***", inner, "*** "));
        wrap_with(will_be_called(), more < less, |inner| print(" ***", inner, "*** "));
    } else {
        wrap_with(will_not_be_called(), true, |inner| print("wrapped result is: ", inner, ""));
    }
}
