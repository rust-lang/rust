trait StaticDefaultRef: 'static {
    fn default_ref() -> &'static Self;
}

impl StaticDefaultRef for str {
    fn default_ref() -> &'static str {
        ""
    }
}

fn into_impl(x: &str) -> &(impl ?Sized + AsRef<str> + StaticDefaultRef + '_) {
    x
}

fn extend_lifetime<'a>(x: &'a str) -> &'static str {
    let t = into_impl(x);
    helper(|_| t) //~ ERROR lifetime may not live long enough
}

fn helper<T: ?Sized + AsRef<str> + StaticDefaultRef>(f: impl FnOnce(&T) -> &T) -> &'static str {
    f(T::default_ref()).as_ref()
}

fn main() {
    let r;
    {
        let x = String::from("Hello World?");
        r = extend_lifetime(&x);
    }
    println!("{}", r);
}
