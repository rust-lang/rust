fn main() {
    fn as_buf<T>(s: str, f: block(str) -> T) -> T { f(s) }
    as_buf("foo", {|foo: str| -> () log_full(core::error, foo);});
}
