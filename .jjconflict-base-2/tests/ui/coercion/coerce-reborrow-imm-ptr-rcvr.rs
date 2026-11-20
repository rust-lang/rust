//@ run-pass

struct SpeechMaker {
    speeches: usize
}

impl SpeechMaker {
    pub fn how_many(&self) -> usize { self.speeches }
}

fn foo(speaker: &SpeechMaker) -> usize {
    speaker.how_many() + 33
}

pub fn main() {
    let lincoln = SpeechMaker {speeches: 22};
    assert_eq!(foo(&lincoln), 55);
}
