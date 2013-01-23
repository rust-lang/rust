struct Wizard {
    spells: ~[&static/str]
}

impl Wizard {
    fn cast(&mut self) {
        for self.spells.each |&spell| {
            io::println(spell);
        }
    }
}

fn main() {
    let mut harry = Wizard {
        spells: ~[ "expelliarmus", "expecto patronum", "incendio" ]
    };
    harry.cast();
}
