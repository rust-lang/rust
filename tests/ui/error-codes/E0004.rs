enum Terminator {
    HastaLaVistaBaby,
    TalkToMyHand,
}

fn main() {
    let x = Terminator::HastaLaVistaBaby;

    match x { //~ ERROR E0004
        Terminator::TalkToMyHand => {}
    }
}
