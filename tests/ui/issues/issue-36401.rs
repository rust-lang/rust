// run-pass
#[derive(Debug)]
pub enum Event {
    Key(u8),
    Resize,
    Unknown(u16),
}

static XTERM_SINGLE_BYTES : [(u8, Event); 1] = [(1,  Event::Resize)];

fn main() {
    match XTERM_SINGLE_BYTES[0] {
        (1, Event::Resize) => {},
        ref bad => panic!("unexpected {:?}", bad)
    }
}
