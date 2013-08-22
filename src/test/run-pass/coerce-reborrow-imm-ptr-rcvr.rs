struct SpeechMaker {
    speeches: uint
}

impl SpeechMaker {
    pub fn how_many(&self) -> uint { self.speeches }
}

fn foo(speaker: &SpeechMaker) -> uint {
    speaker.how_many() + 33
}

pub fn main() {
    let lincoln = SpeechMaker {speeches: 22};
    assert_eq!(foo(&lincoln), 55);
}
