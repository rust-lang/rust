//@ run-pass

struct SpeechMaker {
    speeches: usize
}

fn talk(x: &mut SpeechMaker) {
    x.speeches += 1;
}

fn give_a_few_speeches(speaker: &mut SpeechMaker) {

    // Here speaker is reborrowed for each call, so we don't get errors
    // about speaker being moved.

    talk(speaker);
    talk(speaker);
    talk(speaker);
}

pub fn main() {
    let mut lincoln = SpeechMaker {speeches: 22};
    give_a_few_speeches(&mut lincoln);
}
