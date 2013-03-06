struct SpeechMaker {
    speeches: uint
}

pub impl SpeechMaker {
    pure fn how_many(&self) -> uint { self.speeches }
}

fn foo(speaker: &const SpeechMaker) -> uint {
    speaker.how_many() + 33
}

pub fn main() {
    let mut lincoln = SpeechMaker {speeches: 22};
    fail_unless!(foo(&const lincoln) == 55);
}
