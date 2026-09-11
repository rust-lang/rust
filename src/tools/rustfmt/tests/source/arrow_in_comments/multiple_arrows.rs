// rustfmt-style_edition: 2024
fn main() {
    match a {
        _ => // comment with => 
        match b {
            // one goes to =>
            one => {
                println("1");
            }
            // two goes to =>
            two => { println("2"); }
        } 
    }
}
