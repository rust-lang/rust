//@ run-pass

struct SpeechMaker {
    speeches: usize
}

impl SpeechMaker {
    pub fn talk(&mut self) {
        self.speeches += 1;
    }
}

fn give_a_few_speeches(speaker: &mut SpeechMaker) {

    // Here speaker is reborrowed for each call, so we don't get errors
    // about speaker being moved.

    speaker.talk();
    speaker.talk();
    speaker.talk();
}

pub fn main() {
    let mut lincoln = SpeechMaker {speeches: 22};
    give_a_few_speeches(&mut lincoln);
}
