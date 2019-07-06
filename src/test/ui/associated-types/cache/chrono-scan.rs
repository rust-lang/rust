// check-pass

pub type ParseResult<T> = Result<T, ()>;

pub enum Item<'a> {
    Literal(&'a str)
}

pub fn colon_or_space(s: &str) -> ParseResult<&str> {
    unimplemented!()
}

pub fn timezone_offset_zulu<F>(s: &str, colon: F) -> ParseResult<(&str, i32)>
        where F: FnMut(&str) -> ParseResult<&str> {
    unimplemented!()
}

pub fn parse<'a, I>(mut s: &str, items: I) -> ParseResult<()>
        where I: Iterator<Item=Item<'a>> {
    macro_rules! try_consume {
        ($e:expr) => ({ let (s_, v) = try!($e); s = s_; v })
    }
    let offset = try_consume!(timezone_offset_zulu(s.trim_start(), colon_or_space));
    let offset = try_consume!(timezone_offset_zulu(s.trim_start(), colon_or_space));
    Ok(())
}

fn main() {}
