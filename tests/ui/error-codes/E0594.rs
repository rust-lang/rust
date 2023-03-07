static NUM: i32 = 18;

fn main() {
    NUM = 20; //~ ERROR cannot assign to immutable static item `NUM`
}
