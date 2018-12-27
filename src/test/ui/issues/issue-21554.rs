struct Inches(i32);

fn main() {
    Inches as f32;
    //~^ ERROR casting
}
