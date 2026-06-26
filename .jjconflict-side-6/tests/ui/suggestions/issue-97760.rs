pub fn print_values(values: &impl IntoIterator)
where {
    for x in values.into_iter() {
        println!("{x}");
        //~^ ERROR <impl IntoIterator as IntoIterator>::Item` doesn't implement `std::fmt::Display
    }
}

fn main() {}
