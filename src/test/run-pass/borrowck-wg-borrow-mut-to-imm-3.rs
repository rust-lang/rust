struct Wizard {
    spells: ~[&'static str]
}

pub impl Wizard {
    fn cast(&mut self) {
        for self.spells.each |&spell| {
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
