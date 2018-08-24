pub fn main() {
    let thing = "{{ f }}";
    let f = thing.find("{{");

    if f.is_none() {
        println!("None!");
    }
}
