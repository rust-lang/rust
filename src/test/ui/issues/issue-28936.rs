// build-pass (FIXME(62277): could be check-pass?)
pub type Session = i32;
pub struct StreamParser<'a, T> {
    _tokens: T,
    _session: &'a mut Session,
}

impl<'a, T> StreamParser<'a, T> {
    pub fn thing(&mut self) -> bool { true }
}

pub fn parse_stream<T: Iterator<Item=i32>, U, F>(
        _session: &mut Session, _tokens: T, _f: F) -> U
    where F: Fn(&mut StreamParser<T>) -> U { panic!(); }

pub fn thing(session: &mut Session) {
    let mut stream = vec![1, 2, 3].into_iter();

    let _b = parse_stream(session,
                          stream.by_ref(),
                          // replacing the above with the following fixes it
                          //&mut stream,
                          |p| p.thing());

}

fn main() {}
