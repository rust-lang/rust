fn main() {
    format!(concat!("{foo}"));         //~ ERROR: there is no argument named `foo`
    format!(concat!("{ba", "r} {}"), 1);     //~ ERROR: there is no argument named `bar`
}
