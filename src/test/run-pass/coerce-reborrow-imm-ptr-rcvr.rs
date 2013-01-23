struct SpeechMaker {
    speeches: uint
}

impl SpeechMaker {
    pure fn how_many(&self) -> uint { self.speeches }
}

fn foo(speaker: &const SpeechMaker) -> uint {
    speaker.how_many() + 33
}

fn main() {
    let mut lincoln = SpeechMaker {speeches: 22};
    assert foo(&const lincoln) == 55;
}
