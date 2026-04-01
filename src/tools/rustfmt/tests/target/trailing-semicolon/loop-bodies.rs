// rustfmt-edition: 2024
// rustfmt-style_edition: 2027

macro_rules! dummy {
    () => {};
}

fn main() {
    for x in [1] {
        println!();
    }

    while false {
        println!();
    }

    while let Some('x') = None {
        println!();
    }

    loop {
        println!();
    }

    loop {
        dummy! {}
    }

    loop {
        dummy!();
    }

    loop {
        dummy![];
    }
}
