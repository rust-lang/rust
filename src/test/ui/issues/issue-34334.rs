fn main () {
    let sr: Vec<(u32, _, _) = vec![]; //~ ERROR expected one of `,` or `>`, found `=`
    let sr2: Vec<(u32, _, _)> = sr.iter().map(|(faction, th_sender, th_receiver)| {}).collect();
}
