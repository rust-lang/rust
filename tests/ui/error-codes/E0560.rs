struct Simba {
    mother: u32,
}

fn main() {
    let s = Simba { mother: 1, father: 0 };
    //~^ ERROR E0560
}
