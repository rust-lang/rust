#![allow(unused)]

#[derive(Copy, Clone)]
enum Nucleotide {
    Adenine,
    Thymine,
    Cytosine,
    Guanine
}

#[derive(Clone)]
struct Autosome;

#[derive(Clone)]
enum Allosome {
    X(Vec<Nucleotide>),
    Y(Vec<Nucleotide>)
}

impl Allosome {
    fn is_x(&self) -> bool {
        match *self {
            Allosome::X(_) => true,
            Allosome::Y(_) => false,
        }
    }
}

#[derive(Clone)]
struct Genome {
    autosomes: [Autosome; 22],
    allosomes: (Allosome, Allosome)
}

fn find_start_codon(strand: &[Nucleotide]) -> Option<usize> {
    let mut reading_frame = strand.windows(3);
    // (missing parentheses in `while let` tuple pattern)
    while let b1, b2, b3 = reading_frame.next().expect("there should be a start codon") {
        //~^ ERROR unexpected `,` in pattern
        // ...
    }
    None
}

fn find_thr(strand: &[Nucleotide]) -> Option<usize> {
    let mut reading_frame = strand.windows(3);
    let mut i = 0;
    // (missing parentheses in `if let` tuple pattern)
    if let b1, b2, b3 = reading_frame.next().unwrap() {
        //~^ ERROR unexpected `,` in pattern
        // ...
    }
    None
}

fn is_thr(codon: (Nucleotide, Nucleotide, Nucleotide)) -> bool {
    match codon {
        // (missing parentheses in match arm tuple pattern)
        Nucleotide::Adenine, Nucleotide::Cytosine, _ => true
        //~^ ERROR unexpected `,` in pattern
        _ => false
    }
}

fn analyze_female_sex_chromosomes(women: &[Genome]) {
    // (missing parentheses in `for` tuple pattern)
    for x, _barr_body in women.iter().map(|woman| woman.allosomes.clone()) {
        //~^ ERROR unexpected `,` in pattern
        // ...
    }
}

fn analyze_male_sex_chromosomes(men: &[Genome]) {
    // (missing parentheses in pattern with `@` binding)
    for x, y @ Allosome::Y(_) in men.iter().map(|man| man.allosomes.clone()) {
        //~^ ERROR unexpected `,` in pattern
        // ...
    }
}

fn main() {
    let genomes = Vec::new();
    // (missing parentheses in `let` pattern)
    let women, men: (Vec<Genome>, Vec<Genome>) = genomes.iter().cloned()
    //~^ ERROR unexpected `,` in pattern
        .partition(|g: &Genome| g.allosomes.0.is_x() && g.allosomes.1.is_x());
}
