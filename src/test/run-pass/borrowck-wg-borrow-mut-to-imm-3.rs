struct Wizard {
    spells: ~[&'static str]
}

impl Wizard {
    pub fn cast(&mut self) {
        for self.spells.iter().advance |&spell| {
            println(spell);
        }
    }
}

pub fn main() {
    let mut harry = Wizard {
        spells: ~[ "expelliarmus", "expecto patronum", "incendio" ]
    };
    harry.cast();
}
