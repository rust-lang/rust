fn suggestion(opt: &mut Option<String>) {
    opt = None; //~ ERROR mismatched types
}

fn no_suggestion(opt: &mut Result<String, ()>) {
    opt = None //~ ERROR mismatched types
}

fn suggestion2(opt: &mut Option<String>) {
    opt = Some(String::new())//~ ERROR mismatched types
}

fn no_suggestion2(opt: &mut Option<String>) {
    opt = Some(42)//~ ERROR mismatched types
}

fn main() {}
