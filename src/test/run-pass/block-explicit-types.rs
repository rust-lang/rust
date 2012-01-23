fn main() {
    fn as_buf<T>(s: str, f: fn(str) -> T) -> T { f(s) }
    as_buf("foo", {|foo: str| -> () log(error, foo);});
}
