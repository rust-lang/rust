// run-rustfix

fn get_vowel_count(string: &str) -> usize {
    string
        .chars()
        .filter(|c| "aeiou".contains(c))
        //~^ ERROR expected a `Fn(char)` closure, found `char`
        .count()
}

fn main() {
    let _ = get_vowel_count("asdf");
}
