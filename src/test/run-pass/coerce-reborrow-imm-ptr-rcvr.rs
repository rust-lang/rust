struct SpeechMaker {
    speeches: uint
}

pub impl SpeechMaker {
    fn how_many(&const self) -> uint { self.speeches }
}

fn foo(speaker: &const SpeechMaker) -> uint {
    speaker.how_many() + 33
}

pub fn main() {
    let mut lincoln = SpeechMaker {speeches: 22};
    assert_eq!(foo(&const lincoln), 55);
}
