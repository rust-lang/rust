struct Wizard {
    spells: ~[&'static str]
}

impl Wizard {
    pub fn cast(&mut self) {
        for &spell in self.spells.iter() {
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
