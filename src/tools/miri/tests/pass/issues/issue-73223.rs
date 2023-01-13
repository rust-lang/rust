fn main() {
    let mut state = State { prev: None, next: Some(8) };
    let path = "/nested/some/more";
    assert_eq!(state.rest(path), "some/more");
}

#[allow(unused)]
struct State {
    prev: Option<usize>,
    next: Option<usize>,
}

impl State {
    fn rest<'r>(&mut self, path: &'r str) -> &'r str {
        let start = match self.next.take() {
            Some(v) => v,
            None => return "",
        };

        self.prev = Some(start);
        &path[start..]
    }
}
