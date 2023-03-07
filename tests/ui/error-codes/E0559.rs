enum Field {
    Fool { x: u32 },
}

fn main() {
    let s = Field::Fool { joke: 0 };
    //~^ ERROR E0559
}
