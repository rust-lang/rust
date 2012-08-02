type clam = { chowder: &int };

trait get_chowder {
    fn get_chowder() -> &self/int;
}

impl clam of get_chowder for clam {
    fn get_chowder() -> &self/int { return self.chowder; }
}

fn main() {
    let clam = { chowder: &3 };
    log(debug, *clam.get_chowder());
    clam.get_chowder();
}

