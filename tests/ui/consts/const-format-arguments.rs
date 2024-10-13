
pub fn main() {
    const A: std::fmt::Arguments = std::fmt::Arguments::new_const(&[&"hola"]);
    //~^ use of unstable library feature
}
