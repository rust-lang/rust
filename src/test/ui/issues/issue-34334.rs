fn main () {
    let sr: Vec<(u32, _, _) = vec![];
    //~^ ERROR only path types can be used in associated type constraints
    let sr2: Vec<(u32, _, _)> = sr.iter().map(|(faction, th_sender, th_receiver)| {}).collect();
    //~^ ERROR a value of type `Vec<(u32, _, _)>` cannot be built
}
