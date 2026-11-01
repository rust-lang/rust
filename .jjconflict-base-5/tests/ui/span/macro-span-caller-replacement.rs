macro_rules! macro_with_format { () => {
    fn check_5(arg : usize) -> String {
        let s : &str;
        if arg < 5 {
            s = format!("{arg}"); //~ ERROR mismatched types
        } else {
            s = String::new(); //~ ERROR mismatched types
        }
        String::from(s)
    }
}}

fn main() {
    macro_with_format!();
    println!( "{}", check_5(6) );
}
