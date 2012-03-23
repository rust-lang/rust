type clam = { chowder: &int };

impl clam for clam {
    fn get_chowder() -> &self.int { ret self.chowder; }
}

fn main() {
    let clam = { chowder: &3 };
    log(debug, *clam.get_chowder());
    clam.get_chowder();
}

